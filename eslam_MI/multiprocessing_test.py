import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

warnings.filterwarnings("ignore")

# Import MI files
from utils import *
from models import *
from estimators import *
from trainers import *

# Intialize device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = "CPU"
print(f'Device: {device}')



if __name__ == '__main__':
    mp.set_start_method('spawn')

    # Main parameter: How many indepdent processes do you want
    n_processes = 2
    
    data_dir = os.path.join(os.getcwd(), 'data_for_python')
    X, Y, x_labels, y_labels = process_spike_data(os.path.join(data_dir, '2025-03-12_1'), 0.001)
    print(f"Neural Activity (X): {X.shape}")
    print(f"Muscle Activity (Y): {Y.shape}") 
    print("Neuron Labels:", x_labels)
    print("Muscle Labels:", y_labels)

    params = {
        # Optimizer parameters (for training)
        'epochs': 100,
        'batch_size': 128,
        'learning_rate': 5e-4,
        'n_trials': 2,
        'patience': 50,
        'min_delta': 0.001,
        # Critic parameters for DSIB or DVSIB (for the estimator), except embed_dim, which changes with training
        'Nx': X.shape[0],
        'Ny': Y.shape[0],
        'layers': 2,
        'hidden_dim': 256,
        'activation': 'leaky_relu',
        'beta': 512,
        'max_dz': 10, # max value for embed_dim that we search for
        'estimator': 'infonce', # Estimator: infonce or smile_5. See estimators.py for all options
        'mode': 'sep' # Almost always we'll use separable
    }

    # Make dataset into tensors
    X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    dataset = BatchedDataset(X, Y, params['batch_size'])
    full_dataset = create_data_split(dataset, train_fraction=0.95, device=device)

    torch.cuda.empty_cache()

    # First do a vanilla run to get calibrated
    print('-------------- Vanilla run -------------')
    start = time.time()
    this_params = {**params, 'embed_dim': 1} # Choose dz
    mis, mis_test = train_model(DSIB, full_dataset, this_params)
    print(time.time() - start)
    torch.cuda.empty_cache()

    input_queue = mp.Queue()
    results_queue = mp.Queue()
    # Package together iterator of run_trial arguments for every condition
    iter = []
    for dz in range(1, params['max_dz']+1):
        for j in range(params['n_trials']):
            this_params = {**params, 'embed_dim': dz}
            iter.append((
                j, DSIB, input_queue, results_queue, this_params, device
            ))
    # Split iterator up into 

    processes = []
    for i in iter:
        # Set up each model process
        p = mp.Process(target=run_trial, args=i)
        p.start()
        processes.append(p)
        # Put input data into queue
        input_queue.put(dataset)
    
    time.sleep(1)
    
    for p in processes:
        p.join()

    # Collect results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())