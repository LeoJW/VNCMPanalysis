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
from datatools import *

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
    data_dir = os.path.join(main_dir, '..', 'localdata')
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


moths = [
    # "2025-02-25_1",
    # "2025-02-25",
    "2025-03-11",
    "2025-03-12_1",
    "2025-03-20",
    "2025-03-21"
]

params = {
    # Optimizer parameters (for training)
    'epochs': 250,
    'window_size': 0.05,
    'batch_size': 128, # Number of windows estimator processes at any time
    'learning_rate': 5e-3,
    'patience': 50,
    'min_delta': 0.001,
    'eps': 1e-8, # Use 1e-4 if dtypes are float16, 1e-8 for float32 works okay
    'train_fraction': 0.95,
    'n_test_set_blocks': 5, # Number of contiguous blocks of data to dedicate to test set
    'model_cache_dir': model_cache_dir,
    # Critic parameters for the estimator
    'model_func': DSIB, # DSIB or DVSIB
    'layers': 3,
    'hidden_dim': 64,#512,
    'activation': nn.LeakyReLU, #nn.Softplus
    'embed_dim': 6,
    'beta': 512, # Just used in DVSIB
    'estimator': 'infonce', # Estimator: infonce or smile_5. See estimators.py for all options
    'mode': 'sep', # Almost always we'll use separable
    'max_n_batches': 256, # If input has more than this many batches, encoder runs are split up for memory management
}

all_subsets = {}
all_mi = {}
all_embed_dim = {}
for moth in moths:
    empty_cache()
    ds = TimeWindowDataset(os.path.join(data_dir, moth), window_size=0.05, neuron_label_filter=1)
    this_params = {**params, 'X_dim': ds.X.shape[1] * ds.X.shape[2], 'Y_dim': ds.Y.shape[1] * ds.Y.shape[2]}
    subsets, mi, embed_dim = subsample_MI_vary_embed_dim(ds, this_params, np.arange(1,10), embed_range=np.arange(1,15))
    all_subsets[moth] = subsets
    all_mi[moth] = mi
    all_embed_dim[moth] = embed_dim

save_dicts_to_h5([all_subsets, all_mi, all_embed_dim], filename)
print(f'Final results saved to {filename}')