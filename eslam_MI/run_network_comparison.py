import sys
import os
import subprocess

try:
    import numpy as np
except ImportError as e:
    print('numpy not on here, for some reason')
    pass
try:
    import h5py
except ImportError as e:
    print('h5py not on here')
    pass

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('numpy')
install('h5py')

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
    data_dir = os.path.join(main_dir, '..', 'localdata', 'data_for_python')
    tmpdir = os.environ.get('TMPDIR')
    model_cache_dir = os.path.join(tmpdir, 'model_cache')
    os.makedirs(model_cache_dir, exist_ok=True)
    result_dir = os.path.join(data_dir, 'estimation_runs')
    os.makedirs(result_dir, exist_ok=True)
    print('The linux part totally worked')
# Home version
else:
    main_dir = os.getcwd()
    data_dir = os.path.join(main_dir, '..', 'localdata', 'data_for_python')
    model_cache_dir = os.path.join(main_dir, '..', 'model_cache')
    os.makedirs(model_cache_dir, exist_ok=True)
    result_dir = os.path.join(data_dir, 'estimation_runs')
    os.makedirs(result_dir, exist_ok=True)

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
layers_range = np.array([3,4,5])
stride_range = np.array([2,3])
branch_range = ['1', 'all', None, 'allGrowDilation']
repeats_range = np.arange(3)

precision_noise_levels = np.hstack((0, np.logspace(np.log10(period), np.log10(0.04), 20) / period))
n_repeats = 3

neuron = 25
dataset_neuron = BatchedDatasetWithNoise(X[[neuron],:], Y, params['window_size'])
dataset_all = BatchedDatasetWithNoise(X, Y, params['window_size'])

train_ids = {}
precision_curves = {}
models = {}
time_per_epoch = {}

for n_filters, n_layers, n_stride, branch_layout, rep in product(filter_range, layers_range, stride_range, branch_range, repeats_range):
    empty_cache()
    # Reset to zero noise
    dataset_neuron.apply_noise(0)
    dataset_all.apply_noise(0)
    # Make keys
    key_neuron = f'single_filters_{n_filters}_layers_{n_layers}_stride_{n_stride}_layout_{str(branch_layout)}_rep_{rep}'
    key_all = f'all_filters_{n_filters}_layers_{n_layers}_stride_{n_stride}_layout_{str(branch_layout)}_rep_{rep}'
    this_params = {**params, 'branch': branch_layout, 'stride': n_stride, 'n_filters': n_filters, 'layers': n_layers}
    # Train model on single neuron
    print(key_neuron)
    torch.mps.synchronize()
    tic = time.time()
    mis_test, train_id = train_cnn_model_no_eval(dataset_neuron, this_params)
    torch.mps.synchronize()
    thistime = time.time() - tic
    mod = retrieve_best_model(mis_test, this_params, train_id=train_id)
    # Train model on whole dataset
    print(key_all)
    torch.mps.synchronize()
    tic = time.time()
    mis_test_all, train_id_all = train_cnn_model_no_eval(dataset_all, this_params)
    torch.mps.synchronize()
    thistime_all = time.time() - tic
    mod_all = retrieve_best_model(mis_test_all, this_params, train_id=train_id_all)
    with torch.no_grad():
        # Estimate precision on both models
        precision_mi = np.zeros((len(precision_noise_levels), n_repeats))
        precision_mi_all = np.zeros((len(precision_noise_levels), n_repeats))
        for j0,prec_noise_amp in enumerate(precision_noise_levels):
            for j1 in range(n_repeats):
                dataset_neuron.apply_noise(prec_noise_amp)
                precision_mi[j0,j1] = - mod(dataset_neuron.X, dataset_neuron.Y).detach().cpu().numpy()
                dataset_all.apply_noise(prec_noise_amp)
                precision_mi_all[j0,j1] = - mod_all(dataset_all.X, dataset_all.Y).detach().cpu().numpy()
        # Save results
        train_ids[key_neuron] = train_id
        models[key_neuron] = mod
        precision_curves[key_neuron] = precision_mi
        time_per_epoch[key_neuron] = thistime / len(mis_test)
        
        train_ids[key_all] = train_id_all
        models[key_all] = mod_all
        precision_curves[key_all] = precision_mi_all
        time_per_epoch[key_all] = thistime_all / len(mis_test_all)
    break



# Save output
save_dicts_to_hdf5(
    [train_ids, precision_curves, time_per_epoch],
    os.path.join(result_dir, 'network_arch_comparison_' + datetime.today().strftime('%Y-%m-%d') + '.h5')
)