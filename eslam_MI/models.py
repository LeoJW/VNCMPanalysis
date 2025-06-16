import numpy as np
import torch
import torch.nn as nn
from utils import *
from estimators import *
from datatools import *

# Check if CUDA or MPS is running
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

class DSIB(nn.Module):
    def __init__(self, params, baseline_fn=None):
        super(DSIB, self).__init__()
        self.params = params
        
        # Create two encoders from |X/Y| to |Z_X/Y|
        if params['mode'] in ['sep', 'bi']:
            self.encoder_x = mlp(params['X_dim'], params)
            self.encoder_y = mlp(params['Y_dim'], params)
        if params['mode'] == "bi":
            # Add additional layer to the separable if bilinear
            self.bilinear = nn.Linear(params['embed_dim'], params['embed_dim'], bias=False)
        elif params['mode'] == "concat":
            # Create one encoder that takes |X|+|Y| and outputs 1, it has twice the width as the seprable/bilinear
            self.encoder_xy = mlp(
                dim=params['Nx'] + params['Ny'], hidden_dim=params['hidden_dim'], output_dim=1,
                layers=params['layers'], activation=params['activation']
            )
        # Estimator
        self.info = decoder_INFO(params['estimator'], params['mode'], baseline_fn)

    def forward(self, dataX, dataY):
        batch_size = dataX.shape[0]  # Dynamically infer batch size
        # Concat version
        if self.params['mode'] == "concat":
            x_expanded = dataX.repeat(batch_size, 1)  # Repeat batch_size times
            y_expanded = dataY.repeat_interleave(batch_size, dim=0)  # Repeat within batch
            xy_pairs = torch.cat((x_expanded, y_expanded), dim=1)  # Concatenate along feature dim
            scores = self.encoder_xy(xy_pairs)
            lossGout = self.info(scores, None, batch_size)  # scores is in the place of zX
            return -lossGout
        # Otherwise sep or bi
        # If input is too large, run mini batches
        if batch_size > self.params['max_n_batches']:
            zX = torch.zeros(batch_size, self.params['embed_dim'], device=dataX.device)
            zY = torch.zeros(batch_size, self.params['embed_dim'], device=dataX.device)
            for i in range(0, batch_size, self.params['batch_size']):
                end_idx = min(i + self.params['batch_size'], batch_size)
                zX[i:end_idx,:] = self.encoder_x(dataX[i:end_idx,:,:])
                zY[i:end_idx,:] = self.encoder_y(dataY[i:end_idx,:,:])
        else:
            zX = self.encoder_x(dataX)
            zY = self.encoder_y(dataY)
        if self.params['mode'] == "bi":
            # Get the rotated version ready 
            zX = self.bilinear(zX)
        lossGout = self.info(zX, zY, batch_size)
        return -lossGout

class DSIBconv(nn.Module):
    def __init__(self, params, baseline_fn=None):
        super(DSIB, self).__init__()
        self.params = params
        
        # Create two encoders from |X/Y| to |Z_X/Y|
        if params['mode'] in ['sep', 'bi']:
            self.encoder_x = multi_cnn_mlp(params, use_1d_mode=('Nx' in params and params['Nx'] == 1))
            self.encoder_y = multi_cnn_mlp(params, use_1d_mode=('Ny' in params and params['Ny'] == 1))
        if params['mode'] == "bi":
            # Add additional layer to the separable if bilinear
            self.bilinear = nn.Linear(params['embed_dim'], params['embed_dim'], bias=False)
        elif params['mode'] == "concat":
            # Create one encoder that takes |X|+|Y| and outputs 1, it has twice the width as the seprable/bilinear
            self.encoder_xy = mlp(
                dim=params['Nx'] + params['Ny'], hidden_dim=params['hidden_dim'], output_dim=1,
                layers=params['layers'], activation=params['activation']
            )
        # Estimator
        self.info = decoder_INFO(params['estimator'], params['mode'], baseline_fn)

    def forward(self, dataX, dataY):
        batch_size = dataX.shape[0]  # Dynamically infer batch size
        # Concat version
        if self.params['mode'] == "concat":
            x_expanded = dataX.repeat(batch_size, 1)  # Repeat batch_size times
            y_expanded = dataY.repeat_interleave(batch_size, dim=0)  # Repeat within batch
            xy_pairs = torch.cat((x_expanded, y_expanded), dim=1)  # Concatenate along feature dim
            scores = self.encoder_xy(xy_pairs)
            lossGout = self.info(scores, None, batch_size)  # scores is in the place of zX
            return -lossGout
        # Otherwise sep or bi
        # If input is too large, run mini batches
        if batch_size > self.params['max_n_batches']:
            zX = torch.zeros(batch_size, self.params['embed_dim'], device=dataX.device)
            zY = torch.zeros(batch_size, self.params['embed_dim'], device=dataX.device)
            for i in range(0, batch_size, self.params['batch_size']):
                end_idx = min(i + self.params['batch_size'], batch_size)
                zX[i:end_idx,:] = self.encoder_x(dataX[i:end_idx,:,:,:])
                zY[i:end_idx,:] = self.encoder_y(dataY[i:end_idx,:,:,:])
        else:
            zX = self.encoder_x(dataX)
            zY = self.encoder_y(dataY)
        if self.params['mode'] == "bi":
            # Get the rotated version ready 
            zX = self.bilinear(zX)
        lossGout = self.info(zX, zY, batch_size)
        return -lossGout

class DVSIB(nn.Module):
    def __init__(self, params, baseline_fn=None):
        super(DVSIB, self).__init__()
        self.params = params
        
        if params['mode'] in ["sep", "bi"]:
            # Create two encoders from |X/Y| to |Z_X/Y|
            self.encoder_x = var_mlp(params['X_dim'], params)
            self.encoder_y = var_mlp(params['Y_dim'], params)
        if params['mode'] == "bi":
            # Add additional layer to the separable if bilinear
            self.bilinear = nn.Linear(params['embed_dim'], params['embed_dim'], bias=False)
        if params['mode'] == "concat":
            # Create one encoder that takes |X|+|Y| and outputs 1, it has twice the width as the seprable/bilinear
            self.encoder_xy = var_mlp(params['X_dim'] + params['Y_dim'], params)
        # Estimator
        self.info = decoder_INFO(params['estimator'], params['mode'], baseline_fn)

    def forward(self, dataX, dataY):
        batch_size = dataX.shape[0]  # Dynamically infer batch size
        # Concat version
        if self.params['mode'] == "concat":
            x_expanded = dataX.repeat(batch_size, 1)  # Repeat batch_size times
            y_expanded = dataY.repeat_interleave(batch_size, dim=0)  # Repeat within batch
            xy_pairs = torch.cat((x_expanded, y_expanded), dim=1)  # Concatenate along feature dim
            _, _, scores = self.encoder_xy(xy_pairs)
            lossGin = self.encoder_xy.kl_loss
            lossGout = self.info(scores, None, batch_size)  # scores is in the place of zX
            loss = lossGin - self.params['beta']*lossGout
            return [loss, lossGin, lossGout]
        # Otherwise sep or bi
        # If input is too large, run mini batches
        if batch_size > self.params['max_n_batches']:
            zX = torch.zeros(batch_size, self.params['embed_dim'], device=dataX.device)
            zY = torch.zeros(batch_size, self.params['embed_dim'], device=dataX.device)
            for i in range(0, batch_size, self.params['batch_size']):
                end_idx = min(i + self.params['batch_size'], batch_size)
                zX[i:end_idx,:] = self.encoder_x(dataX[i:end_idx,:,:])[2].squeeze()
                zY[i:end_idx,:] = self.encoder_y(dataY[i:end_idx,:,:])[2].squeeze()
        else:
            _, _, zX = self.encoder_x(dataX)
            _, _, zY = self.encoder_y(dataY)
        # Rotated version for bilinear
        if self.params['mode'] == "bi":
            zX = self.bilinear(zX)
        lossGin = self.encoder_x.kl_loss + self.encoder_y.kl_loss
        lossGout = self.info(zX, zY, batch_size)
        loss = lossGin - self.params['beta'] * lossGout
        return [loss, lossGin, lossGout]


class conv_DVSIB(nn.Module):
    def __init__(self, estimator, params, mode="sep", baseline_fn=None):
        super(conv_DVSIB, self).__init__()
        self.mode = mode
        self.params = params
        if mode in ["sep", "bi"]:
            # Create two encoders from |X/Y| to |Z_X/Y|
            self.encoder_x = var_cnn_mlp(
                input_channels=params['input_channels_x'], hidden_dim=params['hidden_dim_x'], output_dim=params['embed_dim'],
                conv_layers=params['conv_layers_x'], fc_layers=params['fc_layers_x'], activation=params['activation_x']
            )
            self.encoder_y = var_cnn_mlp(
                input_channels=params['input_channels_y'], hidden_dim=params['hidden_dim_y'], output_dim=params['embed_dim'],
                conv_layers=params['conv_layers_y'], fc_layers=params['fc_layers_y'], activation=params['activation_y']
            )
        if mode == "bi":
            # Add additional layer to the separable if bilinear
            self.bilinear = nn.Linear(params['embed_dim'], params['embed_dim'], bias=False)
        if mode == "concat":
            # Create one encoder that takes |X|+|Y| and outputs 1, it has max the width as the separable/bilinear
            self.encoder_xy = var_cnn_mlp(
                input_channels=params['input_channels_x'] + params['input_channels_y'], hidden_dim=max(params['hidden_dim_x'],params['hidden_dim_y']), output_dim=1,
                conv_layers=max(params['conv_layers_x'],params['conv_layers_y']), fc_layers=max(params['fc_layers_x'],params['fc_layers_y']), activation=params['activation_x']
            ) # use x activation
        # Estimator
        self.info = decoder_INFO(estimator, mode, baseline_fn)

    def forward(self, dataX, dataY):
        batch_size = dataX.shape[0]  # Dynamically infer batch size
        if self.mode in ["sep", "bi"]:
            _,_,zX = self.encoder_x(dataX) # Samples
            _,_,zY = self.encoder_y(dataY)
            if self.mode == "bi":
                # Get the rotated version ready 
                zX = self.bilinear(zX)
            lossGin=self.encoder_x.kl_loss + self.encoder_y.kl_loss
            mi = self.info(zX, zY, batch_size)
            lossGout = mi
        elif self.mode == "concat":
            x_expanded = dataX.repeat(batch_size, 1)  # Repeat batch_size times
            y_expanded = dataY.repeat_interleave(batch_size, dim=0)  # Repeat within batch
            xy_pairs = torch.cat((x_expanded, y_expanded), dim=1)  # Concatenate along feature dim
            _, _, scores = self.encoder_xy(xy_pairs)
            lossGin=self.encoder_xy.kl_loss
            mi = self.info(scores, None, batch_size)  # scores is in the place of zX
            lossGout = mi
        
        loss = lossGin - self.params['beta']*lossGout
        return [loss, lossGin, lossGout]
    

# if __name__ == '__main__':
#     import sys
#     import os

#     import torch
#     import random
#     import warnings
#     import time

#     import torch.nn as nn
#     import torch.multiprocessing as mp
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import torch.optim as optim

#     from torch.utils.data import Dataset, DataLoader
#     from scipy.ndimage import gaussian_filter1d
#     from tqdm.auto import tqdm
#     from itertools import product

#     warnings.filterwarnings("ignore")

#     # Import MI files
#     from utils import *
#     from models import *
#     from estimators import *
#     from trainers import *
#     from datatools import *

#     main_dir = os.getcwd()
#     data_dir = os.path.join(main_dir, '..', 'localdata')
#     model_cache_dir = os.path.join(data_dir, 'model_cache')

#     params = {
#     # Optimizer parameters (for training)
#     'epochs': 250,
#     'window_size': 'your_mother', # Window of time the estimator operates on, in samples
#     'batch_size': 1024, # Number of windows estimator processes at any time
#     'learning_rate': 5e-3,
#     'patience': 15,
#     'min_delta': 0.001,
#     'eps': 1e-8, # Use 1e-4 if dtypes are float16, 1e-8 for float32 works okay
#     'train_fraction': 0.9,
#     'model_cache_dir': model_cache_dir,
#     # Critic parameters for the estimator
#     'model_func': DSIB, # DSIB or DVSIB
#     'layers': 3,
#     'hidden_dim': 128,#512,
#     'activation': nn.LeakyReLU, #nn.Softplus
#     'embed_dim': 6,#10,
#     'beta': 512,
#     'estimator': 'infonce', # Estimator: infonce or smile_5. See estimators.py for all options
#     'mode': 'sep', # Almost always we'll use separable
#     'max_n_batches': 256, # If input has more than this many batches, encoder runs are split up for memory management
# }

#     ds = TimeWindowDataset(os.path.join(data_dir, '2025-03-11'), window_size=0.05, neuron_label_filter=1, select_x=[10])

#     this_params = {**params, 'X_dim': ds.X.shape[1] * ds.X.shape[2], 'Y_dim': ds.Y.shape[1] * ds.Y.shape[2], 'model_func': DVSIB}
#     mis_test_dvsib, train_id = train_cnn_model_no_eval(ds, this_params)
#     model_DVSIB = retrieve_best_model(mis_test_dvsib, this_params, train_id=train_id, remove_all=True)