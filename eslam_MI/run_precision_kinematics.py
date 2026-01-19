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
from itertools import product

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
    filename = os.path.join(result_dir, datetime.today().strftime('%Y-%m-%d') + '_kinematics_precision_' + machine + '_' + f'task_{task_id}' + '.h5')
# Otherwise just a single run
else:
    task_id = '0'
    filename = os.path.join(result_dir, datetime.today().strftime('%Y-%m-%d') + '_kinematics_precision_' + machine + '_' + '.h5')
# If file exists add hour to filename
if os.path.isfile(filename):
    filename = filename[:-3] + '_hour_' + datetime.today().strftime('%H') + '.h5'

# Define worker function
def train_models_worker(chunk_with_id):
    params = {
        # Optimizer parameters (for training)
        'epochs': 400,
        # 'window_size': 0.05,
        'batch_size': 1024, # Number of windows estimator processes at any time
        # 's_per_batch': 10, # Alternatively specify seconds of data a batch should be
        'learning_rate': 5e-3,
        'patience': 150,
        'min_delta': 0.001,
        'eps': 1e-8, # Use 1e-4 if dtypes are float16, 1e-8 for float32 works okay
        'train_fraction': 0.95,
        'n_test_set_blocks': 5, # Number of contiguous blocks of data to dedicate to test set
        'epochs_till_max_shift': 20, # Number of epochs until random shifting is at max
        'start_shifting_epoch': 10,
        'model_cache_dir': model_cache_dir,
        # Critic parameters for the estimator
        'model_func': DSIB, # DSIB or DVSIB
        'activation': nn.LeakyReLU,
        'use_bias': False, # Whether to use bias on first layer
        'layers': 4,
        'hidden_dim': 256,
        'embed_dim': 10,
        'beta': 512, # Just used in DVSIB
        'estimator': 'infonce', # Estimator: infonce or smile_5. See estimators.py for all options
        'mode': 'sep', # Almost always we'll use separable
        'max_n_batches': 1024, # If input has more than this many batches, encoder runs are split up for memory management
    }
    embed_repeats = 2
    embed_dims = np.array([4, 8, 12])
    embed_dim_mat = np.vstack([embed_dims for i in range(embed_repeats)])
    window_size_range = np.linspace(0.02, 0.2, 30)

    process_id, chunk = chunk_with_id
    results = []
    for condition in chunk:
        synchronize()
        tic = time.time()
        # Unpack chunk
        muscles, moth = condition
        # Enforce types (fuck python)
        moth = str(moth)
        # Make condition key (hideous but it works)
        mothstring_no_underscore = moth.replace('_','-')
        musclestring = '-'.join(muscles)
        key = f'muscle_{musclestring}_moth_{mothstring_no_underscore}_pid_{process_id}'
        print(f'Process {process_id} key {key}')

        # -------- Step 0: Check that there's data for this muscle/neuron, load dataset
        has_muscles = check_label_present(os.path.join(data_dir, moth), muscles) # Right now this skips running on all muscles
        if np.all(np.logical_not(has_muscles)):
            continue
        # Load dataset
        wi = 5 # Window size to use, close to Manduca wbf
        ds = TimeWindowDatasetKinematics(os.path.join(data_dir, moth), window_size_range[wi], 
            select_x=[0], # Just load one neuron so things run faster
            select_y=muscles,
            only_flapping=True, angles_only=True,
            use_ISI=False, use_phase=True)
        
        # -------- Step 1: Try 3 different embed_dims at intermediate window size, pick best one
        embed_mi = np.zeros(embed_dim_mat.shape, dtype=np.float32)
        embed_model_paths = np.empty(embed_dim_mat.shape, dtype=object)
        for i,embed in enumerate(embed_dims):
            for j in range(embed_repeats):
                this_params = {**params, 
                    'X_dim': ds.Y.shape[1] * ds.Y.shape[2], 'Y_dim': ds.Z.shape[1] * ds.Z.shape[2],
                    'moth': moth,
                    'window_size': window_size_range[wi],
                    'muscles': muscles,
                    'embed_dim' : embed
                }
                # Train model, keep only best one based on early stopping
                ds.move_data_to_windows(time_offset=0)
                mi_test, train_id = train_model_no_eval(ds, this_params, X='Y', Y='Z', verbose=False)
                model_path = retrieve_best_model_path(mi_test, this_params, train_id=train_id, remove_others=True)
                # Just use max test information as MI estimate to save time
                embed_mi[j,i] = mi_test[np.argmax(gaussian_filter1d(np.nan_to_num(mi_test), sigma=1))]
                embed_model_paths[j,i] = model_path
        # Choose smallest embed_dim within 10% of max mutual information value
        embed_mi[embed_mi < 0] = np.nan
        mean_embed_mi = np.nanmean(embed_mi, axis=0)
        max_mi_threshold = np.nanmax(mean_embed_mi) * 0.9
        chosen_embed_ind = np.argmax(mean_embed_mi > max_mi_threshold)
        chosen_rep_ind = np.argmax(embed_mi[:,chosen_embed_ind])
        chosen_embed = embed_dims[chosen_embed_ind]
        # Remove all but chosen model
        for modpath in embed_model_paths[embed_model_paths != embed_model_paths[chosen_rep_ind, chosen_embed_ind]]:
            os.remove(modpath)

        # -------- Step 2: Run over different window sizes
        model_paths = np.empty(window_size_range.shape, dtype=object)
        # One window size was already done, so throw that in
        model_paths[wi] = embed_model_paths[chosen_rep_ind, chosen_embed_ind]
        # Run over the rest of the window sizes
        for i in np.delete(np.arange(len(window_size_range)), wi):
            ds = TimeWindowDatasetKinematics(os.path.join(data_dir, moth), window_size_range[i], 
                select_x=[0], # Just load one neuron so things run faster
                select_y=muscles,
                only_flapping=True, angles_only=True,
                use_ISI=False, use_phase=True
            )
            this_params = {**params, 
                'X_dim': ds.Y.shape[1] * ds.Y.shape[2], 'Y_dim': ds.Z.shape[1] * ds.Z.shape[2],
                'moth': moth,
                'window_size': window_size_range[i],
                'embed_dim': chosen_embed,
                'muscles': muscles
            }
            # Train models
            mi_test, train_id = train_model_no_eval(ds, this_params, X='Y', Y='Z', verbose=False)
            model_paths[i] = retrieve_best_model_path(mi_test, this_params, train_id=train_id, remove_others=True)

        # Save results
        results.append({key : [model_paths, window_size_range, embed_mi, this_params]})
        synchronize()
        print(f'Neuron/muscle condition {key} took {time.time() - tic}')
    return results

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # ------------------------ SETUP ------------------------
    # Main options: How many processes to run in training, how often to save, etc
    # NOTE: MUST BE CALLED ON SLURM WITH N_TASKS OR NOT ALL CONDITIONS WILL BE RUN
    n_tasks = 6
    n_processes = 10
    save_every_n_iterations = 20
    precision_levels = np.logspace(np.log10(0.0001), np.log10(0.2), 400)

    # Package together main iterators
    moth_range = ['2025-02-25', '2025-02-25_1']
    main_iterator = list(product(
        [['lax'], ['lba'], ['lsa'], ['ldvm'], ['ldlm'], ['rdlm'], ['rdvm'], ['rsa'], ['rba'], ['rax'], # Single muscles
        ['lax', 'lba', 'lsa'], # L steering
        ['rax', 'rba', 'rsa'], # R steering
        ['ldvm', 'ldlm'], # L power
        ['rdvm', 'rdlm'], # R power
        ['lax', 'lba', 'lsa', 'rsa', 'rba', 'rax'], # All steering
        ['ldvm', 'ldlm', 'rdlm', 'rdvm'], # All power
        ['lax', 'lba', 'lsa', 'ldvm', 'ldlm', 'rdlm', 'rdvm', 'rsa', 'rba', 'rax'] # All muscles, 
        ],
        moth_range
    ))
    
    # Split into chunks for n tasks, then chunks for n processes
    inds = np.arange(len(main_iterator))
    task_chunk_inds = np.array_split(inds, n_tasks)[task_id]
    chunk_inds = np.array_split(task_chunk_inds, n_processes)
    # chunks_with_ids = [(i, chunk) for i,chunk in enumerate(chunks)]
    chunks = [(ii,[main_iterator[i] for i in inds]) for ii,inds in enumerate(chunk_inds)]

    # ------------------------ TRAINING ------------------------
    # Use multiprocessing pool to train up models
    with mp.Pool(n_processes) as pool:
        train_results = pool.map(train_models_worker, chunks)
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
    precision_noise_curves = {}
    embed_mi_dict = {}
    all_params = {}
    # Run inference serially on resulting models (as inference can take advantage of whole GPU)
    iteration_count = 0
    for key, single_result in results.items():
        # Unpack this result. For some reason types don't change coming out of processes like they do going in
        model_paths, window_size_range, embed_mi, cond_params = single_result
        # model_path, this_params, subsets, mi_subsets = single_result

        # Skip if this moth doesn't have data for all muscles in this moth/neuron/muscle combination
        has_muscles = check_label_present(os.path.join(data_dir, cond_params['moth']), cond_params['muscles'])
        if np.all(np.logical_not(has_muscles)):
            continue
        # Get embed dim used for this neuron/muscle pair
        embed_dim = cond_params['embed_dim']

        # Loop over window sizes, run inference for all, choose winner
        zero_rounding_mi = np.zeros(window_size_range.shape, dtype=np.float32)
        for i,window_size in enumerate(window_size_range):
            new_key = key + f'_window_{window_size}_embed_{embed_dim}'
            print(new_key)
            
            ds = TimeWindowDatasetKinematics(os.path.join(data_dir, cond_params['moth']), cond_params['window_size'], 
                select_x=[0], # Just load one neuron so things run faster
                select_y=cond_params['muscles'],
                only_flapping=True, angles_only=True,
                use_ISI=False, use_phase=True
            )
            this_params = {**cond_params, 
                'X_dim': ds.Y.shape[1] * ds.Y.shape[2], 'Y_dim': ds.Z.shape[1] * ds.Z.shape[2]}
            # Load model, run inference tasks
            with torch.no_grad():
                model = this_params['model_func'](this_params).to(device)
                model.load_state_dict(torch.load(model_paths[i], weights_only=True, map_location=device))
                model.eval()
                precision_mi = precision_rounding(precision_levels, ds, model, X='Y', Y='Z', early_stop=True, early_stop_threshold=0.5)
                precision_noise = precision(precision_levels, ds, model, n_repeats=10, X='Y', Y='Z', early_stop=True, early_stop_threshold=0.5)
                del model
            zero_rounding_mi[i] = precision_mi[0]
            precision_curves[new_key] = precision_mi
            precision_noise_curves[new_key] = precision_noise
            precision_levels_dict[new_key] = precision_levels
            embed_mi_dict[new_key] = embed_mi.flatten()
            all_params[new_key] = [k + ' : ' + str(val) for k, val in this_params.items()]

            empty_cache()
            iteration_count += 1
            if (iteration_count % save_every_n_iterations == 0):
                try:
                    save_dicts_to_h5([precision_levels_dict, precision_curves, precision_noise_curves, all_params, embed_mi_dict], filename)
                    print(f"Intermediate results saved")
                except Exception as e:
                    print(f"Warning: Failed to save intermediate results: {e}")

    # Save final output
    try:
        save_dicts_to_h5([precision_levels_dict, precision_curves, precision_noise_curves, all_params, embed_mi_dict], filename)
        print(f'Final results saved to {filename}')
    except Exception as e:
        print(f"Error saving final results: {e}")
        print("Intermediate files should still be available")