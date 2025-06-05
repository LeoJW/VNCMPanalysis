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

filename = os.path.join(result_dir, 'network_arch_comparison_PACE_' + datetime.today().strftime('%Y-%m-%d') + '.h5')

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

# Read the data and apply some binning/downsampling
period = 0.0001
X, Y, x_labels, y_labels = read_spike_data(os.path.join(data_dir, '2025-03-21'), period)

print(f"Neurons (X): {X.shape}")
print(f"Muscles (Y): {Y.shape}") 
print("Neuron Labels:", x_labels)
print("Muscle Labels:", y_labels)


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
    'branch': '1', # Whether to have branched first layer '1', all branched layers 'all', or None if no branch layers
    'stride': 1, # stride of CNN layers. First layer will always be stride=1
    'n_filters': 32, # Number of new filters per layer. Each layer will 2x this number
    'layers': 4,
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


filter_range = np.array([8, 16, 32])
layers_range = np.array([3,4,5,6,7])
stride_range = np.array([2])
branch_range = [None, '1', 'all', 'allGrowDilation']
repeats_range = np.arange(6)

precision_noise_levels = np.hstack((0, np.logspace(np.log10(period), np.log10(0.06), 100) / period))
n_repeats = 3

neuron = 25
dataset_neuron = BatchedDatasetWithNoise(X[[neuron],:], Y, params['window_size'])
dataset_all = BatchedDatasetWithNoise(X, Y, params['window_size'])

precision_curves = {}
precision_noise = {}
time_per_epoch = {}
all_params = {}

main_iterator = product(["neuron", "all"], filter_range, layers_range, stride_range, branch_range, repeats_range)
iteration_count = 0
save_every_n_iterations = 5
for run_on, n_filters, n_layers, n_stride, branch_layout, rep in main_iterator:
    empty_cache()
    if run_on == "neuron":
        dataset = dataset_neuron
    else:
        dataset = dataset_all
    # Reset to zero noise
    dataset.apply_noise(0)
    # Make keys
    key = f'neuron_{run_on}_filters_{n_filters}_layers_{n_layers}_stride_{n_stride}_layout_{str(branch_layout)}_rep_{rep}'
    this_params = {**params, 'branch': branch_layout, 'stride': n_stride, 'n_filters': n_filters, 'layers': n_layers}

    iteration_count += 1
    print(f"Iteration {iteration_count}, {key}")

    # Train model on whole dataset
    print(key)
    synchronize()
    tic = time.time()
    mis_test, train_id = train_cnn_model_no_eval(dataset, this_params)
    synchronize()
    thistime = time.time() - tic
    model = retrieve_best_model(mis_test, this_params, train_id=train_id, remove_all=True)
    noise_levels, precision_mi = precision(precision_noise_levels, dataset, model, n_repeats=n_repeats)
    # Save results
    precision_curves[key] = precision_mi
    precision_noise[key] = noise_levels
    time_per_epoch[key] = thistime / len(mis_test)
    all_params[key] = [key + ' : ' + str(value) for key, value in this_params.items()]

    if (iteration_count % save_every_n_iterations == 0):
        try:
            save_dicts_to_h5([precision_curves, time_per_epoch, precision_noise, all_params], filename)
            print(f"Intermediate results saved")
        except Exception as e:
            print(f"Warning: Failed to save intermediate results: {e}")


# Save final output
try:
    save_dicts_to_h5([precision_noise, precision_curves, time_per_epoch, all_params], filename)
    print(f'Final results saved to {filename}')
except Exception as e:
    print(f"Error saving final results: {e}")
    print("Intermediate files should still be available")