import os
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
from datatools import *

# Check if CUDA or MPS is running
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# Train functions specific to CNN model architecture
def train_cnn_model(full_dataset, params, device=device, subset_indices=None):
    """
    Generalized training function for DSIB and DVSIB models with early stopping.
    This version does run evaluation! Run time is slower as a result, but more straightforward
    Args:
        model: The model to train (DSIB or DVSIB).
        data: Tuple of (train, test, eval) dataloaders. 
            Assumes uses BatchSubsetDataset with custom sampler
            Assumes loaders return X, Y of shapes (M_x, N) and (M_y, N)
    Returns:
        An array test_estimates containing mutual information estimates of TEST SET ONLY
    """
    # Initialize model
    model_name = params['model_func'].__name__
    model = params['model_func'](params).to(device)
    # Create training dataloader, get indices for test and eval sets
    train_loader, test_indices, eval_indices = create_data_split(full_dataset, params['batch_size'], params['train_fraction'], subset_indices=subset_indices)
    # Pull out test, eval data. Not using a loader as model just needs to do single pass over block of data
    test_X, test_Y = full_dataset.X[test_indices,:,:,:], full_dataset.Y[test_indices,:,:,:]
    eval_X, eval_Y = full_dataset.X[eval_indices,:,:,:], full_dataset.Y[eval_indices,:,:,:]
    # Initialize variables
    epochs = params['epochs']
    opt = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], eps=params['eps'])
    estimates_mi_test = []
    estimates_mi_train = []
    best_estimator_ts = float('-inf')  # Initialize with negative infinity
    no_improvement_count = 0
    for epoch in range(epochs):
        for x, y in iter(train_loader):
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
        # Test model at every epoch
        with torch.no_grad():
            if model_name == "DSIB":
                estimator_tr = -model(eval_X, eval_Y)
                estimator_ts = -model(test_X, test_Y)
            elif model_name == "DVSIB": # Get lossGout, that is the mi value
                _, _, estimator_ts = model(test_X, test_Y)
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
    return np.array(estimates_mi_train), np.array(estimates_mi_test), model


def train_cnn_model_no_eval(full_dataset, params, device=device, subset_indices=None):
    """
    Generalized training function for DSIB and DVSIB models with early stopping.
    Version that does not run evaluation! Skimps on that to save time, returns only mi values from test
    Args:
        full_dataset: (BatchedDataset) with X and Y data
        params: (Dict) with all model, save/load, and training parameters
    Returns:
        estimates_mi_test: Array of mutual information estimates of TEST SET ONLY
        train_id: Name that saved models for each epoch will be found under
    """
    # Initialize model
    model_name = params['model_func'].__name__
    model = params['model_func'](params).to(device)
    # Make save directory if it doesn't exist, generate unique model id
    os.makedirs(params['model_cache_dir'], exist_ok=True)
    train_id = model_name + '_' + f'dz-{params["embed_dim"]}_' + f'bs-{params["window_size"]}_' + str(uuid.uuid4())
    # Create training dataloader, get indices for test set
    train_loader, test_indices, _ = create_data_split(full_dataset, params['batch_size'], params['train_fraction'], subset_indices=subset_indices)
    test_X, test_Y = full_dataset.X[test_indices,:,:], full_dataset.Y[test_indices,:,:]
    # Initialize variables
    epochs = params['epochs']
    opt = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], eps=params['eps'])
    estimates_mi_test = []
    best_estimator_ts = float('-inf')  # Initialize with negative infinity
    no_improvement_count = 0
    for epoch in range(epochs):
        for x, y in iter(train_loader):
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
        # Test model at every epoch
        with torch.no_grad():
            if model_name == "DSIB":
                estimator_ts = -model(test_X, test_Y)
            elif model_name == "DVSIB": # Get lossGout, that is the mi value
                _, _, estimator_ts = model(test_X, test_Y)
            estimator_ts = estimator_ts.to('cpu').detach().numpy()
            estimates_mi_test.append(estimator_ts)
        print(f"Epoch: {epoch+1}, {model_name}, test: {estimator_ts}", flush=True)
        # Save snapshot of model
        torch.save(
            model.state_dict(), 
            os.path.join(params['model_cache_dir'], f'epoch{epoch}_' + train_id + '.pt')
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


def retrieve_best_model(mi_test, params, 
        train_id=None, remove_others=True, remove_all=False, smooth=True, sigma=1, device=device):
    """
    Given a directory for a training run and test MI/loss, loads the model of the best epoch
    Deletes the rest of the model files
    Parameters:
        mi_test (np vector): Vector of test MI at each epoch 
        params (dict): Dictionary of standard model parameters, should match params used during training run. Used for creating model
            Should include model_cache_dir!
        train_id (string): Timestamp unique identifier for that training run. Inferred if not provided
    """
    # Get unique training ids
    valid_files = [f for f in os.listdir(params['model_cache_dir']) if '.pt' in f]
    unique_ids = [s.split('_')[-1][:-3] for s in valid_files]
    # If no time_id provided, use newest
    if train_id is None:
        creation_times = np.array([os.path.getmtime(os.path.join(params['model_cache_dir'], file)) for file in valid_files])
        train_id = unique_ids[np.argmin(creation_times)]
    mi_test = np.nan_to_num(mi_test)
    if smooth:
        mi_test = gaussian_filter1d(mi_test, sigma=sigma)
    # Grab best epoch
    best_epoch = np.argmax(mi_test)
    # Get all files associated with this id, load good one and trash the rest
    match_files = [f for f in valid_files if train_id in f]
    good_file_idx = [idx for idx,s in enumerate(match_files) if f'epoch{best_epoch}' in s][0]
    good_file = match_files.pop(good_file_idx)
    if remove_others or remove_all:
        for file in match_files:
            os.remove(os.path.join(params['model_cache_dir'], file))
    # Determine model class, instantiate and load
    model = params['model_func'](params).to(device)
    model.load_state_dict(torch.load(os.path.join(params['model_cache_dir'], good_file), weights_only=True, map_location=device))
    model.eval()
    if remove_all:
        os.remove(os.path.join(params['model_cache_dir'], good_file))
    return model



# def run_trial(j, model_func, input_queue, results_queue, params, device):
#     """
#     Run a single trial j (and possibly other conditions in opt_params, critic_params, etc)
    
#     Parameters:
#         j (int): Trial index.
#         input_queue: multiprocessing queue to retrieve dataset from
#         results_queue: multiprocessing queue to send result MI values to
#         params (dict): Optimization and critic parameters
#         device: Device to use (e.g., 'cuda', 'mps', 'cpu').
#     Returns:
#         tuple: Results for this trial (key, mis, mis_test).
#     """
#     # Set device explicitly for this process (?)
#     # torch.cuda.empty_cache()  # Clear cache before starting
#     torch.device(device)
#     # Get dataset from queue, create dataloader for train, split out eval and test
#     dataset = input_queue.get()
#     full_dataset = create_data_split(dataset, train_fraction=0.95, device=device)
#     # Initialize and train the model
#     mis, mis_test = train_model(model_func, full_dataset, params)
#     # Return results as a tuple
#     results_queue.put((f"trial_{j}_dz_{params['embed_dim']}", mis, mis_test))
#     # return f"trial_{j}_dz_{dz}", mis, mis_test
