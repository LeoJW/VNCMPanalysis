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
    synchronize = torch.cuda.synchronize
    empty_cache = torch.cuda.empty_cache
elif torch.backends.mps.is_available():
    device = 'mps'
    synchronize = torch.mps.synchronize
    empty_cache = torch.mps.empty_cache
else:
    device = 'cpu'
    synchronize = lambda: None
    empty_cache = lambda: None


"""
Utility functions, lifted from: 
https://stackoverflow.com/questions/51918580/python-random-list-of-numbers-in-a-range-keeping-with-a-minimum-distance
"""
def ranks(sample):
    """
    Return the ranks of each element in an integer sample.
    """
    indices = sorted(range(len(sample)), key=lambda i: sample[i])
    return sorted(indices, key=lambda i: indices[i])
def sample_with_minimum_distance(n=40, k=4, d=10):
    """
    Sample of k elements from range(n), with a minimum distance d.
    """
    sample = np.random.choice(range(n-(k-1)*(d-1)), k, replace=False)
    return np.array([s + (d-1)*r for s, r in zip(sample, ranks(sample))])




def train_model_no_eval(dataset, params, 
        device=device, 
        subset_times=None, 
        return_indices=False, 
        verbose=True,
        X='X', Y='Y' # What to calculate information between. Can be X, Y, or Z
    ):
    """
    Generalized training function for DSIB and DVSIB models with early stopping.
    Version that does not run evaluation! Skimps on that to save time, returns only mi values from test
    Assumes zero offset has been applied to dataset to start with
    Args:
        dataset: (BatchedDataset) with X and Y data
        params: (Dict) with all model, save/load, and training parameters
        subset_times: (array len 2) Times in sec to subset between
        return_indices: (Bool) Whether to return indices for train+test set
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
    # Set up batch size if not provided
    # Uses nearest power of 2 that gives default 10 seconds of data per batch
    # if 's_per_batch' not in params:
    #     params['s_per_batch'] = 10
    if 'batch_size' not in params:
        params['batch_size'] = int(2**np.round(np.log2(params['s_per_batch'] / params['window_size'])))

    # Deal with subsetting
    if subset_times is not None:
        # Turn subset times into window indices
        subset_inds = np.searchsorted(dataset.window_times[dataset.valid_windows], subset_times) - 1
        if subset_inds[1] > subset_inds[0]:
            # Normal case, start time before end time
            num = subset_inds[1] - subset_inds[0]
            use_ind = np.arange(subset_inds[0],subset_inds[1])
        else:
            # Case when times wrap around 
            num = (dataset.n_windows - subset_inds[0]) + subset_inds[1]
            use_ind = (subset_inds[0] + np.arange(0,num)) % dataset.n_windows
        n_windows = num
    else:
        n_windows = dataset.n_windows
        use_ind = np.arange(n_windows)
    # Generate window indices for training and test sets
    n_test_windows = np.floor((1 - params['train_fraction']) * n_windows).astype(int)
    block_len, remainder = divmod(n_test_windows, params['n_test_set_blocks'])
    start_inds = sample_with_minimum_distance(
        n_windows - (block_len + 1), # - (block_len + 1) avoids having block roll past last window
        params['n_test_set_blocks'], 
        np.ceil(n_test_windows / params['n_test_set_blocks']).astype(int)
    )
    # test_block_inds has row 0 start, row 1 end indices
    remainder_split = np.hstack((np.ones(remainder), np.zeros(params['n_test_set_blocks'] - remainder))).astype(int)
    test_block_inds = np.vstack((start_inds, start_inds + remainder_split + block_len))
    # Get time windows from indices, will use later as indices can move around when time offsets applied
    test_block_times = np.vstack((
        dataset.window_times[dataset.valid_windows][use_ind][test_block_inds[0,:]],
        dataset.window_times[dataset.valid_windows][use_ind][test_block_inds[1,:]]
    ))
    # Set arrays of indices for train/test, send to device
    test_indices = np.concatenate([np.arange(test_block_inds[0,i], test_block_inds[1,i]) for i in range(test_block_inds.shape[1])])
    train_indices = np.delete(np.arange(0, n_windows), test_indices)
    # Apply subsetting if needed
    if subset_times is not None:
        test_indices = use_ind[test_indices]
        train_indices = use_ind[train_indices]
    test_indices = torch.tensor(test_indices, dtype=int).to(device)
    train_indices = torch.tensor(train_indices, dtype=int).to(device)
    test_X = getattr(dataset, X)[test_indices,:,:].detach().clone()
    test_Y = getattr(dataset, Y)[test_indices,:,:].detach().clone()

    # Initialize variables
    epochs = params['epochs']
    opt = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], eps=params['eps'])
    estimates_mi_test = []
    best_estimator_ts = float('-inf')  # Initialize with negative infinity
    no_improvement_count = 0

    # Loop over epochs
    for epoch in range(epochs):

        # Shuffle training indices for this epoch, train on batches
        shuffled_train_indices = train_indices[torch.randperm(train_indices.nelement())]
        for batch_indices in shuffled_train_indices.split(params['batch_size']):
            x = getattr(dataset, X)[batch_indices, :, :]
            y = getattr(dataset, Y)[batch_indices, :, :]
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
            # Shuffle time offset of data each epoch
            # Offsets applied after 5 epochs, and offset amount linearly increases to window size after 20 epochs
            if params['epochs_till_max_shift'] == 0:
                max_offset = dataset.window_size
            else:
                max_offset = np.clip(np.abs(epoch - 5) / params['epochs_till_max_shift'], 0, 1) * dataset.window_size
            dataset.move_data_to_windows(time_offset=np.random.uniform(high=max_offset))
            # Get new training set indices (things shift around after applying time offset)
            test_block_inds = np.searchsorted(dataset.window_times[dataset.valid_windows], test_block_times) - 1
            # If subsetting, generate new subset indices from subset times
            if subset_times is not None:
                subset_inds = np.searchsorted(dataset.window_times[dataset.valid_windows], subset_times) - 1
                if subset_inds[1] > subset_inds[0]:
                    # Normal case, start time before end time
                    num = subset_inds[1] - subset_inds[0]
                    use_ind = np.arange(subset_inds[0],subset_inds[1])
                    test_block_inds = test_block_inds - subset_inds[0]
                else:
                    # Case when times wrap around 
                    num = (dataset.n_windows - subset_inds[0]) + subset_inds[1]
                    use_ind = (subset_inds[0] + np.arange(0,num)) % dataset.n_windows
                    test_block_inds = dataset.n_windows - np.abs(test_block_inds - subset_inds[0])
                n_windows = num
            else:
                n_windows = dataset.n_windows
                use_ind = np.arange(n_windows)
            # This excludes windows on either side of test times, to ensure no overlaps with test set
            test_indices = np.concatenate([np.arange(test_block_inds[0,i], test_block_inds[1,i]) for i in range(test_block_inds.shape[1])])
            train_indices = np.delete(np.arange(0, n_windows), test_indices)
            if subset_times is not None:
                train_indices = use_ind[train_indices]
            train_indices = torch.tensor(train_indices, dtype=int).to(device)
        
        if verbose:
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
    
    if return_indices:
        # Set time offset back to zero
        dataset.move_data_to_windows(time_offset=0)
        # If subsetting, generate new subset indices from subset times
        if subset_times is not None:
            subset_inds = np.searchsorted(dataset.window_times[dataset.valid_windows], subset_times) - 1
            if subset_inds[1] > subset_inds[0]:
                # Normal case, start time before end time
                num = subset_inds[1] - subset_inds[0]
                use_ind = np.arange(subset_inds[0],subset_inds[1])
            else:
                # Case when times wrap around 
                num = (dataset.n_windows - subset_inds[0]) + subset_inds[1]
                use_ind = (subset_inds[0] + np.arange(0,num)) % dataset.n_windows
            n_windows = num
        # This excludes windows on either side of test times, to ensure no overlaps with test set
        train_indices = np.arange(0, n_windows)
        if subset_times is not None:
            train_indices = use_ind[train_indices]
        train_indices = torch.tensor(train_indices, dtype=int).to(device)
        
        return np.array(estimates_mi_test), train_id, train_indices
    else:
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

def retrieve_best_model_path(mi_test, params, 
        train_id=None, remove_others=True, smooth=True, sigma=1):
    """
    Given a directory for a training run and test MI/loss, removes all but the best model based on early stopping
    Version of retrieve_best_model that only returns path to good model file
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
    if remove_others:
        for file in match_files:
            os.remove(os.path.join(params['model_cache_dir'], file))
    good_file_path = os.path.join(params['model_cache_dir'], good_file)
    return good_file_path
