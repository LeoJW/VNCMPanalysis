import os
import re
import time
import uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime
from scipy.ndimage import gaussian_filter1d
from utils import *

# Check if CUDA or MPS is running
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = "cpu"

# Train function with early stopping
def train_model(model_func, full_dataset, params, device=device):
    """
    Generalized training function for DSIB and DVSIB models with early stopping.
    Args:
        model: The model to train (DSIB or DVSIB).
        data: Tuple of (train, test, eval) dataloaders. 
            Assumes uses BatchSubsetDataset with custom sampler
            Assumes loaders return X, Y of shapes (M_x, N) and (M_y, N)
    Returns:
        A tuple (train_estimates, test_estimates) containing mutual information estimates.
    """
    # Initialize model
    model_name = model_func.__name__
    model = model_func(params)
    model.to(device)  # Ensure model is on GPU
    # Pull out data loaders
    train_data, (test_X, test_Y), (eval_X, eval_Y) = full_dataset
    # Initialize variables
    epochs = params['epochs']
    opt = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], eps=params['eps'])
    estimates_mi_train = []
    estimates_mi_test = []
    best_estimator_ts = float('-inf')  # Initialize with negative infinity
    no_improvement_count = 0

    for epoch in range(epochs):
        start = time.time()
        for i, (x, y) in enumerate(iter(train_data)):
            # Squeeze to remove batch dim. BatchedDataset handles batches, so it's always just 1
            x, y = x.squeeze(dim=0).T.to(device), y.squeeze(dim=0).T.to(device)
            opt.zero_grad()
            # Compute loss based on model type
            if model_name == "DSIB":
                loss = model(x, y)  # DSIB returns a single loss
            elif model_name == "DVSIB":
                loss, _, _ = model(x, y)  # DVSIB returns three outputs
            else:
                raise ValueError("Invalid model_type. Choose 'DSIB' or 'DVSIB'.")
            loss.backward()
            opt.step()
        print(f'Train time = {time.time() - start}')
        # Evaluate the model at every epoch
        with torch.no_grad():
            if model_name == "DSIB":
                start = time.time()
                estimator_tr = -model(eval_X, eval_Y)
                print(f'Eval time = {time.time() - start}')
                start = time.time()
                estimator_ts = -model(test_X, test_Y)
                print(f'Test time = {time.time() - start}')
            elif model_name == "DVSIB": # Get lossGout, that is the mi value
                start = time.time()
                _, _, estimator_tr = model(eval_X, eval_Y)
                print(f'Eval time = {time.time() - start}')
                start = time.time()
                _, _, estimator_ts = model(test_X, test_Y)
                print(f'Test time = {time.time() - start}')
            estimator_tr = estimator_tr.to('cpu').detach().numpy()
            estimator_ts = estimator_ts.to('cpu').detach().numpy()
            estimates_mi_train.append(estimator_tr)
            estimates_mi_test.append(estimator_ts)
        print(f"Epoch: {epoch+1}, {model_name}, train: {estimator_tr}, test: {estimator_ts}", flush=True)
        # Check for improvement, negative values, or nans
        if np.isnan(estimator_tr) and np.isnan(estimator_ts):
            print('Early stop due to nan outputs')
            break
        if estimator_ts < 0:
            no_improvement_count += 1
        elif estimator_ts > best_estimator_ts + params['min_delta']:
            # We have an improvement
            best_estimator_ts = estimator_ts
            no_improvement_count = 0
        else:
            # No significant improvement
            no_improvement_count += 1
        # Check if we should stop early
        if no_improvement_count >= params['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs. Best estimator_ts: {best_estimator_ts}")
            break
    
    return np.array(estimates_mi_train), np.array(estimates_mi_test)



def train_model_no_eval(model_func, full_dataset, params, model_save_dir, device=device):
    """
    Generalized training function for DSIB and DVSIB models with early stopping.
    Version that does not run evaluation! Skimps on that to save time, returns only mi values from test
    Args:
        model: The model to train (DSIB or DVSIB).
        data: Tuple of (train, test, eval) dataloaders. 
            Assumes uses BatchSubsetDataset with custom sampler
            Assumes loaders return X, Y of shapes (M_x, N) and (M_y, N)
    Returns:
        An array test_estimates containing mutual information estimates of TEST SET ONLY
    """
    # Initialize model
    model_name = model_func.__name__
    model = model_func(params)
    model.to(device)  # Ensure model is on GPU
    # Make save directory if it doesn't exist, generate unique model id
    os.makedirs(model_save_dir, exist_ok=True)
    train_id = model_name + '_' + f'dz-{params["embed_dim"]}_' + f'bs-{params["batch_size"]}_' + str(uuid.uuid4())
    # Pull out data loaders
    train_data, (test_X, test_Y), _ = full_dataset
    # Initialize variables
    epochs = params['epochs']
    opt = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], eps=params['eps'])
    estimates_mi_test = []
    best_estimator_ts = float('-inf')  # Initialize with negative infinity
    no_improvement_count = 0
    for epoch in range(epochs):
        start = time.time()
        for i, (x, y) in enumerate(iter(train_data)):
            # Squeeze to remove batch dim. BatchedDataset handles batches, so it's always just 1
            x, y = x.squeeze(dim=0).T.to(device), y.squeeze(dim=0).T.to(device)
            opt.zero_grad()
            # Compute loss based on model type
            if model_name == "DSIB":
                loss = model(x, y)  # DSIB returns a single loss
            elif model_name == "DVSIB":
                loss, _, _ = model(x, y)  # DVSIB returns three outputs
            else:
                raise ValueError("Invalid model_type. Choose 'DSIB' or 'DVSIB'.")
            loss.backward()
            opt.step()
        print(f'Train time = {time.time() - start}')
        # Test model at every epoch
        with torch.no_grad():
            if model_name == "DSIB":
                start = time.time()
                estimator_ts = -model(test_X, test_Y)
                print(f'Test time = {time.time() - start}')
            elif model_name == "DVSIB": # Get lossGout, that is the mi value
                _, _, estimator_ts = model(test_X, test_Y)
            estimator_ts = estimator_ts.to('cpu').detach().numpy()
            estimates_mi_test.append(estimator_ts)
        print(f"Epoch: {epoch+1}, {model_name}, test: {estimator_ts}", flush=True)
        # Save snapshot of model
        torch.save(
            model, 
            os.path.join(model_save_dir, f'epoch{epoch}_' + train_id + '.pt')
        )
        # Check for improvement, negative values, or nans
        if np.isnan(estimator_ts):
            print('Early stop due to nan outputs')
            break
        if estimator_ts < 0:
            no_improvement_count += 1
        elif estimator_ts > best_estimator_ts + params['min_delta']:
            # We have an improvement
            best_estimator_ts = estimator_ts
            no_improvement_count = 0
        else:
            # No significant improvement
            no_improvement_count += 1
        # Check if we should stop early
        if no_improvement_count >= params['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs. Best estimator_ts: {best_estimator_ts}")
            break
    return np.array(estimates_mi_test), train_id


# def extract_unique_ids_by_time(directory_path):
#     """Quick function to extract unique timestamps from model cache directory"""
#     # Pattern to match the timestamp part in filename format
#     # This will capture the datetime part: 'Thu_24-04-25_20-42-11'
#     pattern = r'.*?([a-zA-Z]{3}_\d{2}-\d{2}-\d{2,4}_\d{2}-\d{2}-\d{2}).*?\.pt$'
#     # Dictionary to store unique timestamps and their datetime objects
#     unique_timestamps = {}
#     valid_files = []
#     # Iterate through all files in the directory
#     for filename in os.listdir(directory_path):
#         match = re.search(pattern, filename)
#         if match:
#             valid_files.append(filename)
#             timestamp_str = match.group(1)
#             # Parse the timestamp string into a datetime object
#             # Format: '%a_%d/%m/%y_%H-%M-%S'
#             try:
#                 dt_obj = datetime.strptime(timestamp_str, '%a_%d-%m-%y_%H-%M-%S')
#                 unique_timestamps[timestamp_str] = dt_obj
#             except ValueError:
#                 # If parsing fails, try with full year format
#                 try:
#                     dt_obj = datetime.strptime(timestamp_str, '%a_%d-%m-%Y_%H-%M-%S')
#                     unique_timestamps[timestamp_str] = dt_obj
#                 except ValueError:
#                     print(f"Could not parse timestamp in file: {filename}")
#     # Sort timestamps by datetime objects (newest first)
#     sorted_timestamps = sorted(unique_timestamps.keys(), 
#         key=lambda x: unique_timestamps[x], 
#         reverse=True)
#     return sorted_timestamps, valid_files

def retrieve_best_model(model_save_dir, mi_test, train_id=None, remove_others=True, burn_in=4, smooth=True, sigma=1):
    """
    Given a directory for a training run and test MI/loss, loads the model of the best epoch
    Deletes the rest of the model files
    Parameters:
        model_save_dir (string/path): Directory model files are stored in
        mi_test (np vector): Vector of test MI at each epoch 
        train_id (string): Timestamp unique identifier for that training run. Inferred if not provided
        burn_in (int): Number of initial epochs to ignore. No way it gets it on the first try, right?
    """
    # Get unique training ids
    valid_files = [f for f in os.listdir(model_save_dir) if '.pt' in f]
    unique_ids = [s.split('_')[-1][:-3] for s in valid_files]
    # If no time_id provided, use newest
    if train_id is None:
        creation_times = np.array([os.path.getmtime(os.path.join(model_save_dir, file)) for file in valid_files])
        train_id = unique_ids[np.argmin(creation_times)]
    mi_test = np.nan_to_num(mi_test)
    if smooth:
        mi_test = gaussian_filter1d(mi_test, sigma=sigma)
    # Grab best epoch
    best_epoch = np.argmax(mi_test[burn_in:]) + burn_in
    # Get all files associated with this id, load good one and trash the rest
    match_files = [f for f in valid_files if train_id in f]
    good_file_idx = [idx for idx,s in enumerate(match_files) if f'epoch{best_epoch}' in s][0]
    good_file = match_files.pop(good_file_idx)
    if remove_others:
        for file in match_files:
            os.remove(os.path.join(model_save_dir, file))
    # Determine model class, instantiate and load
    model = torch.load(os.path.join(model_save_dir, good_file), weights_only=False)
    model.eval()
    return model



def run_trial(j, model_func, input_queue, results_queue, params, device):
    """
    Run a single trial j (and possibly other conditions in opt_params, critic_params, etc)
    
    Parameters:
        j (int): Trial index.
        input_queue: multiprocessing queue to retrieve dataset from
        results_queue: multiprocessing queue to send result MI values to
        params (dict): Optimization and critic parameters
        device: Device to use (e.g., 'cuda', 'mps', 'cpu').
    Returns:
        tuple: Results for this trial (key, mis, mis_test).
    """
    # Set device explicitly for this process (?)
    # torch.cuda.empty_cache()  # Clear cache before starting
    torch.device(device)
    # Get dataset from queue, create dataloader for train, split out eval and test
    dataset = input_queue.get()
    full_dataset = create_data_split(dataset, train_fraction=0.95, device=device)
    # Initialize and train the model
    mis, mis_test = train_model(model_func, full_dataset, params)
    # Return results as a tuple
    results_queue.put((f"trial_{j}_dz_{params['embed_dim']}", mis, mis_test))
    # return f"trial_{j}_dz_{dz}", mis, mis_test
