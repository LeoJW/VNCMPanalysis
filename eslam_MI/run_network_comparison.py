import sys
import os

import torch
import warnings
import time
from datetime import datetime

import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.transforms.functional as TF
import numpy as np
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
    machine = 'PACE'
# Home version
else:
    main_dir = os.getcwd()
    data_dir = os.path.join(main_dir, '..', 'localdata')
    model_cache_dir = os.path.join(data_dir, 'model_cache')
    os.makedirs(model_cache_dir, exist_ok=True)
    result_dir = os.path.join(data_dir, 'estimation_runs')
    os.makedirs(result_dir, exist_ok=True)
    machine = 'HOME'

# Case where script got an input argument, means multiple separate runs
if len(sys.argv) > 1: 
    task_id = sys.argv[1]
    print(f'Task ID is {task_id}')
    filename = os.path.join(result_dir, datetime.today().strftime('%Y-%m-%d') + '_network_comparison_' + machine + '_' + f'task_{task_id}' + '.h5')
# Otherwise just a single run
else:
    filename = os.path.join(result_dir, datetime.today().strftime('%Y-%m-%d') + '_network_comparison_' + machine + '_' + '.h5')
# If file exists add hour to filename
if os.path.isfile(filename):
    filename = filename[:-3] + '_hour_' + datetime.today().strftime('%H') + '.h5'

# Define worker function
def train_models_worker(chunk_with_id):
    params = {
        # Optimizer parameters (for training)
        'epochs': 300,
        # 'window_size': 0.05,
        # 'batch_size': 256, # Number of windows estimator processes at any time
        's_per_batch': 30, # Alternatively specify seconds of data a batch should be
        'learning_rate': 2e-3,
        'patience': 50,
        'min_delta': 0.001,
        'eps': 1e-8, # Use 1e-4 if dtypes are float16, 1e-8 for float32 works okay
        'train_fraction': 0.95,
        'n_test_set_blocks': 5, # Number of contiguous blocks of data to dedicate to test set
        'epochs_till_max_shift': 10, # Number of epochs until random shifting is at max
        'model_cache_dir': model_cache_dir,
        # Critic parameters for the estimator
        'model_func': DSIB, # DSIB or DVSIB
        # 'activation': nn.PReLU,#nn.LeakyReLU,
        # 'use_bias': True, # Whether to use bias on first layer
        # 'layers': 3,
        # 'hidden_dim': 64,
        # 'embed_dim': 6,
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
        run_on, hidden_dim, window_size, layers, embed_dim, use_ISI, use_bias, activation_str, rep = condition
        # Enforce types (fuck python)
        hidden_dim = int(hidden_dim)
        window_size = float(window_size)
        layers = int(layers)
        embed_dim = int(embed_dim)
        use_ISI = True if use_ISI == "True" else False
        use_bias = True if use_bias == "True" else False
        activation = getattr(nn, activation_str)
        rep = int(rep)
        # Make condition key (hideous but it works)
        key = (
            f'neuron_{run_on}_hiddendim_{hidden_dim}_window_{window_size}_layers_{layers}_'
            f'embed_{embed_dim}_ISI_{use_ISI}_bias_{use_bias}_activation_{activation_str}_'
            f'rep_{rep}_pid_{process_id}'
        )
        print(key)
        # Load dataset
        select = [10] if run_on == "neuron" else None
        ds = TimeWindowDataset(os.path.join(data_dir, '2025-03-11'), window_size, 
            neuron_label_filter=1, select_x=select, use_ISI=use_ISI)
        # Set params
        this_params = {**params, 
            'X_dim': ds.X.shape[1] * ds.X.shape[2], 'Y_dim': ds.Y.shape[1] * ds.Y.shape[2],
            'window_size': window_size, 
            'hidden_dim': hidden_dim,
            'layers': layers, 
            'embed_dim': embed_dim,
            'run_on': run_on, 
            'use_ISI': use_ISI,
            'use_bias': use_bias,
            'activation': activation}
        # Train model, keep only best one based on early stopping
        synchronize()
        tic = time.time()
        mi_test, train_id = train_model_no_eval(ds, this_params, verbose=False)
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
    n_processes = 12
    save_every_n_iterations = 20
    precision_levels = np.logspace(np.log10(0.0001), np.log10(1.0), 250)

    # Package together main iterators
    hidden_dim_range = np.array([32, 64, 128, 256, 512])
    window_size_range = np.logspace(np.log10(0.02), np.log10(1.0), 10)
    layers_range = np.array([3,4,5,6])
    embed_dim_range = np.array([1,2,4,8,12])
    use_ISI_range = [True]
    use_bias_range = [False, True]
    activation_range = ["LeakyReLU", "PReLU"]
    repeats_range = np.arange(1)

    main_iterator = product(
        ["neuron", "all"], 
        hidden_dim_range, 
        window_size_range, 
        layers_range, 
        embed_dim_range, 
        use_ISI_range,
        use_bias_range,
        activation_range,
        repeats_range)

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
        # Unpack this result. For some reason types don't change coming out of processes like they do going in
        model_path, this_time_per_epoch, this_params = single_result
        # Load dataset
        select = [10] if this_params['run_on'] == "neuron" else None
        ds = TimeWindowDataset(os.path.join(data_dir, '2025-03-11'), window_size=this_params['window_size'], 
            neuron_label_filter=1, select_x=select, use_ISI=this_params['use_ISI'])
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