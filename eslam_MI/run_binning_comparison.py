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

filename = os.path.join(result_dir, 'binning_PACE_' + datetime.today().strftime('%Y-%m-%d') + '.h5')

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
    "2025-02-25_1",
    "2025-02-25",
    "2025-03-11",
    "2025-03-12_1",
    "2025-03-20",
    "2025-03-21"
]
moth = moths[2]

params = {
    # Optimizer parameters (for training)
    'epochs': 250,
    'window_size': 512, # Window of time the estimator operates on, in samples
    'batch_size': 128, # Number of windows estimator processes at any time
    'learning_rate': 5e-3,
    'patience': 10,
    'min_delta': 0.001,
    'eps': 1e-8, # Use 1e-4 if dtypes are float16, 1e-8 for float32 works okay
    'train_fraction': 0.9,
    'model_cache_dir': model_cache_dir,
    # Critic parameters for the estimator
    'model_func': DSIB, # DSIB or DVSIB
    'branch': 'allGrowDilation', # Whether to have branched first layer '1', all branched layers 'all', or None if no branch layers
    'stride': 2, # stride of CNN layers. First layer will always be stride=1
    'n_filters': 8, # Number of new filters per layer. Each layer will 2x this number
    'layers': 5,
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

n_repeats = 3
neuron = 3

window_len_range = np.array([50, 100])
period_range = np.logspace(np.log10(0.00005), np.log10(0.01), 20)

precision_noise = {}
precision_mi = {}
all_params = {}

main_iterator = product(["neuron", "all"], range(n_repeats), period_range, window_len_range)
iteration_count = 0
total_iterations = len(list(main_iterator))
save_every_n_iterations = 5
for run_on, rep, period, window_len in main_iterator:
    empty_cache()
    precision_noise_levels = np.hstack((0, np.logspace(np.log10(period), np.log10(0.05), 40) / period))
    
    # Make keys
    key = f'neuron_{run_on}_window_{window_len}_period_{period}_rep_{rep}'
    this_params = {**params, 'window_size': np.round(window_len / 1000 / period).astype(int)}

    iteration_count += 1
    print(f"[{iteration_count}/{total_iterations}] {key}")
    
    X, Y, x_labels, y_labels = read_spike_data(os.path.join(data_dir, moth), period)
    if run_on == "neuron":
        dataset = BatchedDatasetWithNoise(X[[neuron],:], Y, this_params['window_size'])
    else:
        dataset = BatchedDatasetWithNoise(X, Y, this_params['window_size'])
    
    # Train models, run precision
    mis_test, train_id = train_cnn_model_no_eval(dataset, this_params)
    model = retrieve_best_model(mis_test, this_params, train_id=train_id, remove_all=True)
    noise_levels, mi = precision(precision_noise_levels, dataset, model, n_repeats=n_repeats)

    precision_noise[key] = noise_levels
    precision_mi[key] = mi
    all_params[key] = this_params

    if (iteration_count % save_every_n_iterations == 0):
        try:
            save_dicts_to_h5([precision_noise, precision_mi, all_params], filename)
            print(f"Intermediate results saved")
        except Exception as e:
            print(f"Warning: Failed to save intermediate results: {e}")

save_dicts_to_h5([precision_noise, precision_mi, all_params], filename)
print(f'Final results saved to {filename}')