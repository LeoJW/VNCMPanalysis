import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Check if CUDA or MPS is running
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = "cpu"

# Train function with early stopping, serial form
def train_model(model, full_dataset, opt_params, model_type="dsib", device=device, patience=50, min_delta=0.001):
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
    model.to(device)  # Ensure model is on GPU
    epochs = opt_params['epochs']
    opt = torch.optim.Adam(model.parameters(), lr=opt_params['learning_rate'])
    estimates_mi_train = []
    estimates_mi_test = []
    # Pull out data loaders
    train_data, (test_X, test_Y), (eval_X, eval_Y) = full_dataset
    # Early stopping variables
    best_estimator_ts = float('-inf')  # Initialize with negative infinity
    no_improvement_count = 0

    for epoch in range(epochs):        
        for i, (x, y) in enumerate(iter(train_data)):
            # Squeeze to remove batch dim. BatchedDataset handles batches, so it's always just 1
            x, y = x.squeeze(dim=0).T.to(device), y.squeeze(dim=0).T.to(device)
            if epoch == 0:
                print(x)
            opt.zero_grad()
            # Compute loss based on model type
            if model_type == "dsib":
                loss = model(x, y)  # DSIB returns a single loss
            elif model_type == "dvsib":
                loss, _, _ = model(x, y)  # DVSIB returns three outputs
            else:
                raise ValueError("Invalid model_type. Choose 'dsib' or 'dvsib'.")
            loss.backward()
            opt.step()
        # Evaluate the model at every epoch
        with torch.no_grad():
            if model_type == "dsib":
                estimator_tr = -model(eval_X, eval_Y)
                estimator_ts = -model(test_X, test_Y)
            elif model_type == "dvsib": # Get lossGout, that is the mi value
                _, _, estimator_tr = model(eval_X, eval_Y)
                _, _, estimator_ts = model(test_X, test_Y)
            estimator_tr = estimator_tr.to('cpu').detach().numpy()
            estimator_ts = estimator_ts.to('cpu').detach().numpy()
            estimates_mi_train.append(estimator_tr)
            estimates_mi_test.append(estimator_ts)
        print(f"Epoch: {epoch+1}, {model_type}, train: {estimator_tr}, test: {estimator_ts}", flush=True)
        
        # Check for improvement or negative values
        if estimator_ts < 0:
            no_improvement_count += 1
        elif estimator_ts > best_estimator_ts + min_delta:
            # We have an improvement
            best_estimator_ts = estimator_ts
            no_improvement_count = 0
        else:
            # No significant improvement
            no_improvement_count += 1
        # Check if we should stop early
        if no_improvement_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best estimator_ts: {best_estimator_ts}")
            break
    
    return np.array(estimates_mi_train), np.array(estimates_mi_test)


def train_model_parallel(full_dataset, model_func, critic_params, opt_params, device=device, patience=50, min_delta=0.001):
    """
    Generalized training function for DSIB and DVSIB models with early stopping.
    Parallelized form, creates DSIB/DVSIB models within function so models are unique to each process 
    Args:
        data: Tuple of (train, test, eval) dataloaders. 
            Assumes uses BatchSubsetDataset with custom sampler
            Assumes loaders return X, Y of shapes (M_x, N) and (M_y, N)
        model_type: Either "dsib" (returns loss, that's negative mi) or "dvsib" (returns loss, lossGin, lossGout).
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
    Returns:
        A tuple (train_estimates, test_estimates) containing mutual information estimates.
    """
    # Pull out data loaders
    train_data, (test_X, test_Y), (eval_X, eval_Y) = full_dataset
    # Initialize model
    model_name = model_func.__name__
    model = model_func(critic_params)
    model.to(device)  # Ensure model is on GPU
    # Initialize variables
    best_estimator_ts = float('-inf')  # Initialize with negative infinity
    no_improvement_count = 0
    epochs = opt_params['epochs']
    opt = torch.optim.Adam(model.parameters(), lr=opt_params['learning_rate'])
    estimates_mi_train = []
    estimates_mi_test = []

    for epoch in range(epochs):        
        for i, (x, y) in enumerate(iter(train_data)):
            # Squeeze to remove batch dim. BatchedDataset handles batches, so it's always just 1
            x, y = x.squeeze(dim=0).T.to(device), y.squeeze(dim=0).T.to(device)
            if epoch == 0:
                print(x)
            opt.zero_grad()
            # Compute loss based on model type
            if model_name == "DSIB":
                loss = model(x, y)  # DSIB returns a single loss
            elif model_name == "DVSIB":
                loss, _, _ = model(x, y)  # DVSIB returns three outputs
            else:
                raise ValueError("Invalid model_type. Choose 'dsib' or 'dvsib'.")
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
        elif estimator_ts > best_estimator_ts + min_delta:
            # We have an improvement
            best_estimator_ts = estimator_ts
            no_improvement_count = 0
        else:
            # No significant improvement
            no_improvement_count += 1
        # Check if we should stop early
        if no_improvement_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best estimator_ts: {best_estimator_ts}")
            break
    
    return np.array(estimates_mi_train), np.array(estimates_mi_test)