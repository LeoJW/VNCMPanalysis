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
    data_dir = os.path.join(main_dir, '..', 'localdata')
    model_cache_dir = os.path.join(data_dir, 'model_cache')
    os.makedirs(model_cache_dir, exist_ok=True)
    result_dir = os.path.join(data_dir, 'estimation_runs')
    os.makedirs(result_dir, exist_ok=True)

# Case where script got an input argument, means multiple separate runs
if len(sys.argv) > 1: 
    task_id = sys.argv[1]
    print(f'Task ID is {task_id}')
    filename = os.path.join(result_dir, datetime.today().strftime('%Y-%m-%d') + '_network_comparison_PACE_' + f'task_{task_id}' + '.h5')
# Otherwise just a single run
else:
    filename = os.path.join(result_dir, datetime.today().strftime('%Y-%m-%d') + '_network_comparison_PACE_' + '.h5')

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

params = {
    # Optimizer parameters (for training)
    'epochs': 250,
    'batch_size': 256, # Number of windows estimator processes at any time
    'learning_rate': 5e-3,
    'n_trials': 3,
    'patience': 15,
    'min_delta': 0.001,
    'eps': 1e-8, # Use 1e-4 if dtypes are float16, 1e-8 for float32 works okay
    'train_fraction': 0.9,
    'model_cache_dir': model_cache_dir,
    # Critic parameters for the estimator
    'model_func': DSIB, # DSIB or DVSIB
    'layers': 4,
    'hidden_dim': 256,
    'activation': nn.LeakyReLU, #nn.Softplus
    'embed_dim': 10,
    'beta': 512,
    'estimator': 'infonce', # Estimator: infonce or smile_5. See estimators.py for all options
    'mode': 'sep', # Almost always we'll use separable
    'max_n_batches': 256, # If input has more than this many batches, encoder runs are split up for memory management
}


layers_range = np.array([3,4,5,6,7])
hidden_dim_range = np.array([128, 256, 512, 1024])
window_size_range = np.logspace(np.log10(0.02), np.log10(2.0), 100)
embed_dim_range = np.array([2,6,10,14])
repeats_range = np.arange(1)

precision_noise_levels = np.hstack((0, np.logspace(np.log10(0.002), np.log10(0.1), 30)))
n_repeats = 50

precision_curves = {}
precision_noise = {}
time_per_epoch = {}
all_params = {}

main_iterator = product(["neuron", "all"], layers_range, hidden_dim_range, window_size_range, embed_dim_range, repeats_range)
iteration_count = 0
save_every_n_iterations = 5
for run_on, n_layers, hidden_dim, window_size, embed_dim, rep in main_iterator:
    empty_cache()
    # Set up dataset
    if run_on == "neuron":
        ds = TimeWindowDataset(os.path.join(data_dir, '2025-03-11'), window_size, neuron_label_filter=1, select_x=[10])
    else:
        ds = TimeWindowDataset(os.path.join(data_dir, '2025-03-11'), window_size, neuron_label_filter=1)
    # Make keys, params
    key = f'neuron_{run_on}_layers_{n_layers}_hiddendim_{hidden_dim}_window_{window_size}_embed_{embed_dim}_rep_{rep}'
    this_params = {**params, 
        'layers': n_layers, 'hidden_dim': hidden_dim, 'window_size': window_size, 'embed_dim': embed_dim,
        'X_dim': ds.X.shape[1] * ds.X.shape[2], 'Y_dim': ds.Y.shape[1] * ds.Y.shape[2]
    }

    iteration_count += 1
    print(f"Iteration {iteration_count}, {key}")

    # Train model on whole dataset
    print(key)
    synchronize()
    tic = time.time()
    mis_test, train_id = train_cnn_model_no_eval(ds, this_params)
    synchronize()
    thistime = time.time() - tic
    model = retrieve_best_model(mis_test, this_params, train_id=train_id, remove_all=True)
    precision_mi = precision(precision_noise_levels, ds, model, n_repeats=n_repeats)
    # Save results
    precision_curves[key] = precision_mi
    precision_noise[key] = precision_noise_levels
    time_per_epoch[key] = thistime / len(mis_test)
    all_params[key] = [key + ' : ' + str(value) for key, value in this_params.items()]

    if (iteration_count % save_every_n_iterations == 0):
        try:
            save_dicts_to_h5([precision_noise, precision_curves, time_per_epoch, all_params], filename)
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