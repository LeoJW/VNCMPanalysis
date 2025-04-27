import numpy as np
import torch
import torch.nn as nn
from utils import *
from estimators import *

# Check if CUDA or MPS is running
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = "cpu"

class DSIB(nn.Module):
    def __init__(self, params, baseline_fn=None):
        super(DSIB, self).__init__()
        self.params = params
        
        if params['mode'] in ["sep", "bi"]:
            # Create two encoders from |X/Y| to |Z_X/Y|
            self.encoder_x = mlp(
                dim=params['Nx'], hidden_dim=params['hidden_dim'], output_dim=params['embed_dim'],
                layers=params['layers'], activation=params['activation']
            )
            self.encoder_y = mlp(
                dim=params['Ny'], hidden_dim=params['hidden_dim'], output_dim=params['embed_dim'],
                layers=params['layers'], activation=params['activation']
            )
            
        if params['mode'] == "bi":
            # Add additional layer to the separable if bilinear
            self.bilinear = nn.Linear(params['embed_dim'], params['embed_dim'], bias=False)

        if params['mode'] == "concat":
            # Create one encoder that takes |X|+|Y| and outputs 1, it has twice the width as the seprable/bilinear
            self.encoder_xy = mlp(
                dim=params['Nx'] + params['Ny'], hidden_dim=params['hidden_dim'], output_dim=1,
                layers=params['layers'], activation=params['activation']
            )
        
        # Estimator
        self.info = decoder_INFO(params['estimator'], params['mode'], baseline_fn)

    def forward(self, dataX, dataY):
        batch_size = dataX.shape[0]  # Dynamically infer batch size
        if self.params['mode'] in ["sep", "bi"]:
            zX = self.encoder_x(dataX)
            zY = self.encoder_y(dataY)
            if self.params['mode'] == "bi":
                # Get the rotated version ready 
                zX = self.bilinear(zX)
            if (self.params['chunked_inference'] and
                (self.params['chunk_size'] < batch_size) and 
                (self.params['estimator'] == 'infonce')):
                mi = estimate_full_mutual_information_infonce(self, zX, zY, chunk_size=self.params['chunk_size'])
            else:
                mi = self.info(zX, zY, batch_size)
            lossGout = mi
        elif self.params['mode'] == "concat":
            x_expanded = dataX.repeat(batch_size, 1)  # Repeat batch_size times
            y_expanded = dataY.repeat_interleave(batch_size, dim=0)  # Repeat within batch
            xy_pairs = torch.cat((x_expanded, y_expanded), dim=1)  # Concatenate along feature dim
            scores = self.encoder_xy(xy_pairs)
            mi = self.info(scores, None, batch_size)  # scores is in the place of zX
            lossGout = mi
        loss = -lossGout
        return loss

class DVSIB(nn.Module):
    def __init__(self, params, baseline_fn=None):
        super(DVSIB, self).__init__()
        self.params = params
        
        if params['mode'] in ["sep", "bi"]:
            # Create two encoders from |X/Y| to |Z_X/Y|
            self.encoder_x = var_mlp(
                dim=params['Nx'], hidden_dim=params['hidden_dim'], output_dim=params['embed_dim'],
                layers=params['layers'], activation=params['activation']
            )
            self.encoder_y = var_mlp(
                dim=params['Ny'], hidden_dim=params['hidden_dim'], output_dim=params['embed_dim'],
                layers=params['layers'], activation=params['activation']
            )
        
        if params['mode'] == "bi":
            # Add additional layer to the separable if bilinear
            self.bilinear = nn.Linear(params['embed_dim'], params['embed_dim'], bias=False)
        
        if params['mode'] == "concat":
            # Create one encoder that takes |X|+|Y| and outputs 1, it has twice the width as the seprable/bilinear
            self.encoder_xy = var_mlp(
                dim=params['Nx'] + params['Ny'], hidden_dim=params['hidden_dim'], output_dim=1,
                layers=params['layers'], activation=params['activation']
            )
        
        # Estimator
        self.info = decoder_INFO(params['estimator'], params['mode'], baseline_fn)

    def forward(self, dataX, dataY):
        batch_size = dataX.shape[0]  # Dynamically infer batch size
        # Chunked version of sep infonce inference, if asked
        if (self.params['chunked_inference'] and
            (self.params['mode'] == 'sep') and 
            (self.params['chunk_size'] < batch_size) and 
            (self.params['estimator'] == 'infonce')):
            # Pre-compute all embeddings before running estimate in chunks
            with torch.no_grad():
                _, _, zX = self.encoder_x(dataX)
                _, _, zY = self.encoder_y(dataY)
            lossGin = self.encoder_x.kl_loss + self.encoder_y.kl_loss
            lossGout = estimate_full_mutual_information_infonce(self, zX, zY, chunk_size=self.params['chunk_size'])
            loss = lossGin - self.params['beta']*lossGout
            return [loss, lossGin, lossGout]

        if self.params['mode'] in ["sep", "bi"]:
            _, _, zX = self.encoder_x(dataX) # Samples
            _, _, zY = self.encoder_y(dataY)
            if self.params['mode'] == "bi":
                # Get the rotated version ready 
                zX = self.bilinear(zX)
            lossGin=self.encoder_x.kl_loss + self.encoder_y.kl_loss
            mi = self.info(zX, zY, batch_size)
            lossGout = mi
        elif self.params['mode'] == "concat":
            x_expanded = dataX.repeat(batch_size, 1)  # Repeat batch_size times
            y_expanded = dataY.repeat_interleave(batch_size, dim=0)  # Repeat within batch
            xy_pairs = torch.cat((x_expanded, y_expanded), dim=1)  # Concatenate along feature dim
            _, _, scores = self.encoder_xy(xy_pairs)
            lossGin=self.encoder_xy.kl_loss
            mi = self.info(scores, None, batch_size)  # scores is in the place of zX
            lossGout = mi
        loss = lossGin - self.params['beta']*lossGout
        return [loss, lossGin, lossGout]


