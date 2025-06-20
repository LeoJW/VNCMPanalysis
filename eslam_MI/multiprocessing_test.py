import sys
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import warnings
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from itertools import islice

warnings.filterwarnings("ignore")

# Import MI files
from utils import *
from models import *
from estimators import *
from trainers import *
from datatools import *

# Intialize device
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

# Define worker function
def train_models_worker(chunk_with_id):
    params = {
        # Optimizer parameters (for training)
        'epochs': 250,
        # 'window_size': 0.05,
        'batch_size': 256, # Number of windows estimator processes at any time
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
        # 'hidden_dim': 64,
        'activation': nn.LeakyReLU,
        'embed_dim': 6,
        'beta': 512, # Just used in DVSIB
        'estimator': 'infonce', # Estimator: infonce or smile_5. See estimators.py for all options
        'mode': 'sep', # Almost always we'll use separable
        'max_n_batches': 256, # If input has more than this many batches, encoder runs are split up for memory management
    }

    process_id, chunk = chunk_with_id
    results = []
    for condition in chunk:
        synchronize()
        chunktic = time.time()
        # Unpack chunk
        hidden_dim, window_size = condition
        # Enforce types (fuck python)
        hidden_dim = int(hidden_dim)
        window_size = float(window_size)
        # Make condition key
        key = f'hiddendim_{hidden_dim}_window_{window_size}_pid_{process_id}'
        print(key)
        # Load dataset
        ds = TimeWindowDataset(os.path.join(data_dir, '2025-03-11'), window_size, neuron_label_filter=1)#, select_x=[10])
        # Set params
        this_params = {**params, 
            'X_dim': ds.X.shape[1] * ds.X.shape[2], 'Y_dim': ds.Y.shape[1] * ds.Y.shape[2],
            'window_size': window_size, 'hidden_dim': hidden_dim}
        # Train model, keep only best one based on early stopping
        synchronize()
        tic = time.time()
        mi_test, train_id = train_model_no_eval(ds, this_params)
        synchronize()
        thistime = (time.time() - tic) / len(mi_test)
        print(f'seconds/epoch = {thistime}')
        model_path = retrieve_best_model_path(mi_test, this_params, train_id=train_id)
        # Save results
        results.append({key : [model_path, thistime, this_params]})
        synchronize()
        print(f'-------------- Chunk took {time.time() - chunktic}')
    return results


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # Main options: How many processes to run in training, how often to save, etc
    n_processes = 5
    save_every_n_iterations = 5
    precision_levels = np.hstack((0, np.logspace(np.log10(0.0001), np.log10(0.15), 100)))

    # Package together main iterator
    # hidden_dim_range = np.array([32, 64, 128, 256, 512])
    # window_size_range = np.logspace(np.log10(0.02), np.log10(1.0), 10)
    hidden_dim_range = np.array([32, 64, 128])
    window_size_range = np.logspace(np.log10(0.05), np.log10(1.0), 20)
    # main_iterator = product(hidden_dim_range, window_size_range)
    main_iterator = product(np.repeat(hidden_dim_range[0], 3), np.repeat(window_size_range[0], 20))
    # Split into chunks, add process id
    chunks = np.array_split(list(main_iterator), n_processes)
    chunks_with_ids = [(i, chunk) for i,chunk in enumerate(chunks)]

    # ------------------------ TRAINING ------------------------
    # Use multiprocessing pool to train up models
    with mp.Pool(n_processes) as pool:
        train_results = pool.map(train_models_worker, chunks_with_ids)
    # Merge into single dict
    results = {}
    for process_output in train_results:
        if not len(process_output) == 0:
            for d in process_output:
                results.update(d)

    # ------------------------ INFERENCE ------------------------
    # Set up dicts to save
    precision_curves = {}
    precision_levels_dict = {}
    time_per_epoch = {}
    all_params = {}
    # Run inference serially on resulting models (as inference can take advantage of whole GPU)
    iteration_count = 0
    for key, single_result in results.items():
        # Unpack this result
        model_path, this_time_per_epoch, this_params = single_result
        # Load dataset
        ds = TimeWindowDataset(os.path.join(data_dir, '2025-03-11'), window_size=this_params['window_size'], neuron_label_filter=1)
        # Load model, run inference tasks
        with torch.no_grad():
            model = this_params['model_func'](this_params).to(device)
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
            model.eval()
            os.remove(model_path)
            precision_mi = precision_rounding(precision_levels, ds, model)
        precision_curves[key] = precision_mi
        precision_levels_dict[key] = precision_levels
        time_per_epoch[key] = this_time_per_epoch
        all_params[key] = [k + ' : ' + str(val) for k, val in this_params.items()]

        iteration_count += 1
        if (iteration_count % save_every_n_iterations == 0):
            try:
                save_dicts_to_h5([precision_levels_dict, precision_curves, time_per_epoch, all_params], filename)
                print(f"Intermediate results saved")
            except Exception as e:
                print(f"Warning: Failed to save intermediate results: {e}")

    # Save final output
    try:
        save_dicts_to_h5([precision_levels_dict, precision_curves, time_per_epoch, all_params], filename)
        print(f'Final results saved to {filename}')
    except Exception as e:
        print(f"Error saving final results: {e}")
        print("Intermediate files should still be available")