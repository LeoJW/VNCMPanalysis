import sys
import os

import torch
import time

import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d
from itertools import product

from utils import *
from models import *
from estimators import *
from trainers import *

# Directory handling
# PACE version
if sys.platform == 'linux':
    main_dir = os.getcwd()
    data_dir = os.path.join(main_dir, '..', '..', 'localdata')
    tmpdir = os.environ.get('TMPDIR')
    model_cache_dir = os.path.join(tmpdir, 'model_cache')
    os.makedirs(model_cache_dir, exist_ok=True)
    result_dir = os.path.join(data_dir, 'estimation_runs')
    os.makedirs(result_dir, exist_ok=True)
# Home version
else:
    main_dir = os.getcwd()
    data_dir = os.path.join(main_dir, '..', '..', 'localdata')
    model_cache_dir = os.path.join(data_dir, 'model_cache')
    os.makedirs(model_cache_dir, exist_ok=True)
    result_dir = os.path.join(data_dir, 'estimation_runs')
    os.makedirs(result_dir, exist_ok=True)

filename = os.path.join(result_dir, 'subsampling_PACE_' + datetime.today().strftime('%Y-%m-%d') + '.h5')

# Set defaults, device
default_dtype = torch.float32
torch.set_default_dtype(default_dtype)
if torch.cuda.is_available():
    device = 'cuda'
    synchronize = torch.cuda.synchronize
    empty_cache = torch.cuda.empty_cache
elif torch.backends.mps.is_available():
    device = 'mps'
    synchronize = torch.mps.synchronize
    empty_cache = torch.mps.empty_cache
else:
    device = "CPU"
    synchronize = lambda: None
    empty_cache = lambda: None
print(f'Device: {device}')

period = 0.0001

moths = [
    "2025-02-25_1",
    "2025-02-25",
    "2025-03-11",
    "2025-03-12_1",
    "2025-03-20",
    "2025-03-21"
]


params = {
    # Optimizer parameters (for training)
    'epochs': 250,
    'window_size': 512, # Window of time the estimator operates on, in samples
    'batch_size': 128, # Number of windows estimator processes at any time
    'learning_rate': 5e-3,
    'n_trials': 3,
    'patience': 10,
    'min_delta': 0.001,
    'eps': 1e-8, # Use 1e-4 if dtypes are float16, 1e-8 for float32 works okay
    'train_fraction': 0.9,
    'model_cache_dir': model_cache_dir,
    # Critic parameters for the estimator
    'model_func': DSIB, # DSIB or DVSIB
    'branch': 'expDilation', # Whether to have branched first layer '1', all branched layers 'all', or None if no branch layers
    'stride': 2, # stride of CNN layers. First layer will always be stride=1
    'n_filters': 8, # Number of new filters per layer. Each layer will 2x this number
    'layers': 7,
    'fc_layers': 2, # fully connected layers
    'hidden_dim': 256,
    'activation': nn.LeakyReLU, #nn.Softplus
    'embed_dim': 10,
    'beta': 512,
    'max_dz': 12, # max value for embed_dim that we search for
    'estimator': 'infonce', # Estimator: infonce or smile_5. See estimators.py for all options
    'mode': 'sep', # Almost always we'll use separable
    'max_n_batches': 256, # If input has more than this many batches, encoder runs are split up for memory management
}

all_subsets = {}
all_mi = {}
all_embed_dim = {}
for moth in moths:
    empty_cache()
    X, Y, x_labels, y_labels = read_spike_data(os.path.join(data_dir, moth), period)
    dataset = BatchedDataset(X, Y, params['window_size'])
    subsets, mi, embed_dim = subsample_MI_vary_embed_dim(dataset, params, np.arange(1,10), embed_range=np.arange(1,15))
    all_subsets[moth] = subsets
    all_mi[moth] = mi
    all_embed_dim[moth] = embed_dim

save_dicts_to_h5([all_subsets, all_mi, all_embed_dim], filename)
print(f'Final results saved to {filename}')