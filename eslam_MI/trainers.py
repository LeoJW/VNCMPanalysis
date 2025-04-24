import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *

# Check if CUDA or MPS is running
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = "cpu"

# Train function with early stopping, serial form
def train_model(model_func, full_dataset, params, device=device):
    """
    Generalized training function for DSIB and DVSIB models with early stopping.
    Args:
        model: The model to train (DSIB or DVSIB).
        data: Tuple of (train, test, eval) dataloaders. 
            Assumes uses BatchSubsetDataset with custom sampler
            Assumes loaders return X, Y of shapes (M_x, N) and (M_y, N)
        model_type: Either "dsib" (returns loss, that's negative mi) or "dvsib" (returns loss, lossGin, lossGout).
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
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
    opt = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    estimates_mi_train = []
    estimates_mi_test = []
    best_estimator_ts = float('-inf')  # Initialize with negative infinity
    no_improvement_count = 0

    for epoch in range(epochs):        
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
        # Evaluate the model at every epoch
        with torch.no_grad():
            if model_name == "DSIB":
                estimator_tr = -model(eval_X, eval_Y)
                estimator_ts = -model(test_X, test_Y)
            elif model_name == "DVSIB": # Get lossGout, that is the mi value
                _, _, estimator_tr = model(eval_X, eval_Y)
                _, _, estimator_ts = model(test_X, test_Y)
            estimator_tr = estimator_tr.to('cpu').detach().numpy()
            estimator_ts = estimator_ts.to('cpu').detach().numpy()
            estimates_mi_train.append(estimator_tr)
            estimates_mi_test.append(estimator_ts)
        print(f"Epoch: {epoch+1}, {model_name}, train: {estimator_tr}, test: {estimator_ts}", flush=True)
        
        # Check for improvement or negative values
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
    full_dataset = create_train_test_eval(dataset, train_fraction=0.95, device=device)
    # Initialize and train the model
    mis, mis_test = train_model(model_func, full_dataset, params)
    # Return results as a tuple
    results_queue.put((f"trial_{j}_dz_{params['embed_dim']}", mis, mis_test))
    # return f"trial_{j}_dz_{dz}", mis, mis_test
