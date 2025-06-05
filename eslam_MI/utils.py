import os
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import product
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from estimators import estimate_mutual_information
from datatools import *
from trainers import *


# Check if CUDA or MPS is running
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = "cpu"

def rho_to_mi(dim, rho):
    """Obtain the ground truth mutual information from rho."""
    return -0.5 * np.log2(1 - rho**2) * dim


def mi_to_rho(dim, mi):
    """Obtain the rho for Gaussian give ground truth mutual information."""
    return np.sqrt(1 - 2**(-2.0 / dim * mi))


def linear_mut_info(x, y, threshold=1e-10):
    try:
        # Combine x and y column-wise (variables are columns)
        xy = np.hstack((x, y))
        
        # Compute joint covariance matrix once
        c_tot = np.cov(xy, rowvar=False)
        n_x = x.shape[1]  # Number of features in X
        n_y = y.shape[1]  # Number of features in Y
        
        # Extract C_x and C_y from the joint covariance matrix
        c_x = c_tot[:n_x, :n_x]
        c_y = c_tot[n_x:, n_x:]
        
        # Compute eigenvalues using eigh (faster for symmetric matrices)
        eig_tot = np.linalg.eigh(c_tot)[0]  # Returns sorted eigenvalues (ascending)
        eig_x = np.linalg.eigh(c_x)[0]
        eig_y = np.linalg.eigh(c_y)[0]
        
        # Threshold eigenvalues (avoid log(0))
        eig_tot_thr = np.maximum(eig_tot, threshold)
        eig_x_thr = np.maximum(eig_x, threshold)
        eig_y_thr = np.maximum(eig_y, threshold)
        
        # Compute log determinants
        logdet_tot = np.sum(np.log2(eig_tot_thr))
        logdet_x = np.sum(np.log2(eig_x_thr))
        logdet_y = np.sum(np.log2(eig_y_thr))
        
        # Mutual information
        info = 0.5 * (logdet_x + logdet_y - logdet_tot)
        return info if not np.isinf(info) else np.nan
    except np.linalg.LinAlgError:
        return np.nan


class mlp(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim, layers, activation):
        """Create an mlp from the configurations."""
        super(mlp, self).__init__()
    
        # Initialize the layers list
        seq = []
    
        # Input layer
        seq.append(nn.Linear(dim, hidden_dim))
        seq.append(activation())
        nn.init.xavier_uniform_(seq[0].weight)  # Xavier initialization for input layer
    
        # Hidden layers
        for _ in range(layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            nn.init.xavier_uniform_(layer.weight)  # Xavier initialization for hidden layers
            seq.append(layer)
            seq.append(activation())
    
        # Connect all together before the output
        self.base_network = nn.Sequential(*seq)
    
        # Output layer
        self.out = nn.Linear(hidden_dim, output_dim)
        
        # Initialize the layer with Xavier initialization
        nn.init.xavier_uniform_(self.out.weight)
    
    def forward(self, x):
        x = self.base_network(x)
        
        # Get output
        out = self.out(x)
        
        return out


class var_mlp(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim, layers, activation):
        """Create a variational mlp from the configurations."""
        super(var_mlp, self).__init__()
        # Initialize the layers list
        seq = []
        # Input layer
        seq.append(nn.Linear(dim, hidden_dim))
        seq.append(activation())
        nn.init.xavier_uniform_(seq[0].weight)  # Xavier initialization for input layer
        # Hidden layers
        for _ in range(layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            nn.init.xavier_uniform_(layer.weight)  # Xavier initialization for hidden layers
            seq.append(layer)
            seq.append(activation())
        # Connect all together before the output
        self.base_network = nn.Sequential(*seq)
        # Two heads for means and log variances
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)
        # Initialize the heads with Xavier initialization
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        # Normal distribution for sampling
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        # KL Divergence loss initialized to zero
        self.kl_loss = 0.0
        # Set limits for numerical stability
        self.logvar_min = -20  # Lower bound for logVar
        self.logvar_max = 20   # Upper bound for logVar

    def forward(self, x):
        x = self.base_network(x)
        # Get mean and log variance
        meanz = self.fc_mu(x)
        logVar = self.fc_logvar(x)
        # Clamp logVar to prevent extreme values
        logVar = torch.clamp(logVar, min=self.logvar_min, max=self.logvar_max)
        # Compute KL divergence loss
        kl_terms = 0.5 * (torch.square(meanz) + torch.exp(logVar) - 1 - logVar)
        self.kl_loss = torch.mean(torch.sum(kl_terms, dim=1))
        # Check for NaN in KL loss
        if torch.isnan(self.kl_loss):
            print("NaN detected in KL loss!")
            # Use a small default value instead of NaN
            self.kl_loss = torch.tensor(0.1, device=device, requires_grad=True)
        # Reparameterization trick
        epsilon = self.N.sample(meanz.shape)
        std = torch.exp(0.5 * logVar)
        samples = meanz + std * epsilon
        return [meanz, logVar, samples]


class cnn_mlp(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_dim, conv_layers, fc_layers, activation, stride):
        """
        Create a CNN-MLP hybrid model with batch normalization and strided convolutions.
        
        Args:
            input_channels (int): Number of input channels (e.g., 3 for RGB images).
            hidden_dim (int): Number of hidden units in fully connected layers.
            output_dim (int): Dimensionality of the output embedding.
            conv_layers (int): Number of convolutional layers.
            fc_layers (int): Number of fully connected layers after GAP.
            activation (str): Activation function to use (e.g., 'relu', 'tanh').
        """
        super(cnn_mlp, self).__init__()
        
        # Convolutional layers with strided convolutions and batch normalization
        conv_seq = []
        in_channels = input_channels
        out_channels = 32  # Start with 32 filters
        for i in range(conv_layers):
            conv_seq.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride if i > 0 else 1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                activation()
            ))
            in_channels = out_channels
            out_channels *= 2  # Double the number of filters in each layer
        self.conv_network = nn.Sequential(*conv_seq)
        # Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layers
        fc_seq = []
        in_features = in_channels  # After GAP, the number of features equals the number of channels
        for _ in range(fc_layers):
            fc_seq.append(nn.Linear(in_features, hidden_dim))
            fc_seq.append(activation())
            in_features = hidden_dim
        self.fc_network = nn.Sequential(*fc_seq)
        # Output layer
        self.out = nn.Linear(in_features, output_dim)
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Pass through convolutional layers
        x = self.conv_network(x)
        # Apply global average pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten to [batch_size, num_features]
        # Pass through fully connected layers
        x = self.fc_network(x)
        # Get output
        out = self.out(x)
        return out


class MultiScaleConvBlock(nn.Module):
    """
    A multi-scale convolutional block with parallel branches of different kernel sizes.
    Captures features at multiple scales and concatenates them.
    """
    def __init__(self, in_channels, out_channels, stride=1, dilation=2, activation=nn.ReLU):
        super(MultiScaleConvBlock, self).__init__()
        # Calculate channels for each branch (distribute output channels more flexibly)
        # Strategy: Give larger branches slightly more channels
        base_channels = out_channels // 2
        remainder = out_channels % 2
        # Distribute remainder channels to branches (prioritize faster branches)
        branch1_channels = base_channels + remainder  # 3x3 conv  
        branch2_channels = base_channels + remainder  # 5x5 conv
        # Branch 1: 3x3 conv (standard receptive field)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch1_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(branch1_channels),
            activation()
        )
        # Branch 2: 3x3 conv with dilation (larger receptive field)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch2_channels, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False),
            nn.BatchNorm2d(branch2_channels),
            activation()
        )
    
    def forward(self, x):
        # Process through all branches in parallel
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        # Concatenate all branches along channel dimension to get out_channels outputs
        return torch.cat([branch1_out, branch2_out], dim=1)

class multi_cnn_mlp(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_dim, conv_layers, fc_layers, activation, stride, n_filters, branch_layout):
        """
        Create a CNN-MLP hybrid model with batch normalization and strided convolutions.
        
        Args:
            input_channels (int): Number of input channels (e.g., 3 for RGB images).
            hidden_dim (int): Number of hidden units in fully connected layers.
            output_dim (int): Dimensionality of the output embedding.
            conv_layers (int): Number of convolutional layers.
            fc_layers (int): Number of fully connected layers after GAP.
            activation (str): Activation function to use (e.g., 'relu', 'tanh').
        """
        super(multi_cnn_mlp, self).__init__()
        
        # Multi-scale convolutional layers with strided convolutions and batch normalization
        conv_seq = []
        in_channels = input_channels
        out_channels = n_filters
        # Please re-write this to not be insane
        for i in range(conv_layers):
            if branch_layout is None:
                conv_seq.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1 if i == 0 else stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    activation()
                ))
            elif branch_layout == '1':
                if i == 0:
                    conv_seq.append(MultiScaleConvBlock(in_channels, out_channels, stride=1, activation=activation))
                else:
                    conv_seq.append(nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1 if i == 0 else stride, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        activation()
                    ))
            elif branch_layout == 'all':
                conv_seq.append(MultiScaleConvBlock(in_channels, out_channels, stride=1 if i == 0 else stride, activation=activation))
            else:
                conv_seq.append(MultiScaleConvBlock(in_channels, out_channels, stride=1 if i == 0 else stride, dilation=i+1, activation=activation))
            # Double the number of filters in each layer
            in_channels = out_channels
            out_channels *= 2
        self.conv_network = nn.Sequential(*conv_seq)
        # Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layers
        fc_seq = []
        in_features = in_channels  # After GAP, the number of features equals the number of channels
        for _ in range(fc_layers):
            fc_seq.append(nn.Linear(in_features, hidden_dim))
            fc_seq.append(activation())
            in_features = hidden_dim
        self.fc_network = nn.Sequential(*fc_seq)
        # Output layer
        self.out = nn.Linear(in_features, output_dim)
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Pass through branches of convolutional layers
        x = self.conv_network(x)
        # Concatenate along channel dimension, apply global average pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten to [batch_size, num_features]
        # Pass through fully connected layers
        x = self.fc_network(x)
        # Get output
        out = self.out(x)
        return out


class var_cnn_mlp(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_dim, conv_layers, fc_layers, activation):
        """
        Create a variational CNN-MLP hybrid model from the configurations.
        
        Args:
            input_channels (int): Number of input channels (e.g., 3 for RGB images).
            hidden_dim (int): Number of hidden units in fully connected layers.
            output_dim (int): Dimensionality of the latent space (mean and log variance).
            conv_layers (int): Number of convolutional layers.
            fc_layers (int): Number of fully connected layers after GAP.
            activation (str): Activation function to use (e.g., 'relu', 'tanh').
        """
        super(var_cnn_mlp, self).__init__()
        
        # Define activation functions
        activation_fn = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            'silu': nn.SiLU,
        }[activation]
        
        # Convolutional layers
        conv_seq = []
        in_channels = input_channels
        out_channels = 32  # Start with 32 filters
        for _ in range(conv_layers):
            conv_seq.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            conv_seq.append(activation_fn())
            conv_seq.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Downsample spatial dimensions
            in_channels = out_channels
            out_channels *= 2  # Double the number of filters in each layer
        self.conv_network = nn.Sequential(*conv_seq)
        # Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layers
        fc_seq = []
        in_features = in_channels  # After GAP, the number of features equals the number of channels
        for _ in range(fc_layers):
            fc_seq.append(nn.Linear(in_features, hidden_dim))
            fc_seq.append(activation_fn())
            in_features = hidden_dim
        self.fc_network = nn.Sequential(*fc_seq)
        # Two heads for mean and log variance
        self.fc_mu = nn.Linear(in_features, output_dim)
        self.fc_logvar = nn.Linear(in_features, output_dim)
        # Initialize weights
        self._initialize_weights()
        # Normal distribution for sampling
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)  # Move to the appropriate device
        self.N.scale = self.N.scale.to(device)
        # KL Divergence loss initialized to zero
        self.kl_loss = 0.0
        # Set limits for numerical stability
        self.logvar_min = -20  # Lower bound for logVar
        self.logvar_max = 20   # Upper bound for logVar

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # Pass through convolutional layers
        x = self.conv_network(x)
        # Apply global average pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten to [batch_size, num_features]
        # Pass through fully connected layers
        x = self.fc_network(x)
        # Get mean and log variance
        meanz = self.fc_mu(x)
        logVar = self.fc_logvar(x)
        # Clamp logVar to prevent extreme values
        logVar = torch.clamp(logVar, min=self.logvar_min, max=self.logvar_max)
        # Compute KL divergence loss
        kl_terms = 0.5 * (torch.square(meanz) + torch.exp(logVar) - 1 - logVar)
        self.kl_loss = torch.mean(torch.sum(kl_terms, dim=1))
        # Check for NaN in KL loss
        if torch.isnan(self.kl_loss):
            print("NaN detected in KL loss!")
            # Use a small default value instead of NaN
            self.kl_loss = torch.tensor(0.1, device=device, requires_grad=True)
        # Reparameterization trick
        epsilon = self.N.sample(meanz.shape)
        std = torch.exp(0.5 * logVar)
        samples = meanz + std * epsilon
        return [meanz, logVar, samples]


class decoder_INFO(nn.Module):
    def __init__(self, typeEstimator, mode="sep", baseline_fn=None):
        super(decoder_INFO, self).__init__()
        
        self.estimator = typeEstimator
        self.baseline_fn = baseline_fn
        self.mode = mode  # "sep" and "bi" use the same critic function

    def critic_fn(self, dataZX, dataZY, batch_size=None):
        if self.mode in ["sep", "bi"]:  
            return torch.matmul(dataZY, dataZX.t())
        elif self.mode == "concat":
            return torch.reshape(dataZX, [batch_size, batch_size]).t() # Here dataZX is really the final scores matrix
        else:
            raise ValueError("Invalid mode. Choose 'sep', 'bi', or 'concat'.")

    def forward(self, dataZX, dataZY, batch_size=None):
        return estimate_mutual_information(
            self.estimator, 
            self.critic_fn(dataZX, dataZY, batch_size),
            log_baseline=self.baseline_fn
        )

def write_config(args):
    out_fn = "config.json"
    out_fp = os.path.join(args.save_dir, out_fn)
    with open(out_fp, 'w') as fh:
        json.dump(vars(args), fh)

def log_prob_gaussian(x):
    return torch.sum(torch.distributions.Normal(0., 1.).log_prob(x), -1)



def subsample_MI(dataset, params, split_sizes=np.arange(1,6)):
    """
    Subsamples dataset into non-overlapping fractions, trains estimators on all fractions
    This version does not vary dimensionality of embedding space (embed_dim)
    Args:
        dataset: (BatchedDataset) Input data
        params: (Dict) Dict of params for model, training, etc 
    Returns:
        subsets: (np.array) Vector of how many subsamples a given MI value belongs to
        mi: (np.array) Vector of MI values
    """
    # Generate random subsample indices
    indices = []
    for ss in split_sizes:
        inds = np.random.choice(dataset.X.shape[0], dataset.X.shape[0], replace=False)
        edges = np.rint(np.linspace(0, dataset.X.shape[0]-1, ss+1)).astype(int)
        for i in range(ss):
            indices.append(inds[edges[i]:edges[i+1]])
    mi = np.zeros((len(indices)))
    subsets = np.hstack([np.repeat(x, x) for x in split_sizes]) # run length encoding of split_sizes
    # Loop over subsets
    for i,inds in enumerate(indices):
        # Train model
        mis_test, train_id = train_cnn_model_no_eval(dataset, params, subset_indices=inds)
        mod = retrieve_best_model(mis_test, params, train_id=train_id, remove_all=True)
        # Run model inference to get MI value
        with torch.no_grad():
            mi[i] = - mod(dataset.X[inds,:,:,:], dataset.Y[inds,:,:,:]).detach().cpu().numpy()
    return subsets, mi


def subsample_MI_vary_embed_dim(dataset, params, split_sizes=[1,2,3,4,5,6], embed_range=np.arange(2, 15)):
    """
    Subsamples dataset into non-overlapping fractions, trains estimators on all fractions
    This version DOES vary dimensionality of embedding space (embed_dim)
    Args:
        dataset: (BatchedDataset) Input data
        params: (Dict) Dict of params for model, training, etc 
    Returns:
        subsets: (np.array) Vector of how many subsamples a given MI value belongs to
        mi: (np.array) Vector of MI values
        embed_dim_vec: (np.array) Vector of embedding dims
    """
    # Generate random subsample indices
    indices = []
    for ss in split_sizes:
        inds = np.random.choice(dataset.X.shape[0], dataset.X.shape[0], replace=False)
        edges = np.rint(np.linspace(0, dataset.X.shape[0]-1, ss+1))
        for i in range(ss):
            indices.append(inds[edges[i]:edges[i+1]])
    mi, embed_dim_vec = [], []
    subsets = np.hstack([np.repeat(x, x) for x in split_sizes]) # run length encoding of split_sizes
    # Loop over subsets
    for subset_idx, embed_dim in product(range(len(subsets)), embed_range):
        inds = indices[subset_idx]
        this_params = {**params, 'embed_dim': embed_dim}
        # Train model
        mis_test, train_id = train_cnn_model_no_eval(dataset, this_params, subset_indices=inds)
        with torch.no_grad():
            # Retrieve model, run inference to get MI value
            mod = retrieve_best_model(mis_test, this_params, train_id=train_id, remove_all=True)
            mi.append(- mod(dataset.X[inds,:,:,:], dataset.Y[inds,:,:,:]).detach().cpu().numpy())
            embed_dim_vec.append(embed_dim)
    mi = np.array(mi)
    embed_dim_vec = np.array(embed_dim_vec)
    return subsets, mi, embed_dim_vec


def precision(noise_levels, dataset, model, n_repeats=3):
    """
    Run spike timing precision analysis, to get precision curve
    """
    with torch.no_grad():
        # Since datasets are discrete samples, only run noise levels that are actually unique (in no. of samples)
        _, unique_inds = np.unique(np.round(noise_levels), return_index=True)
        new_noise_levels = noise_levels[unique_inds]
        mi = np.zeros((len(new_noise_levels), n_repeats))
        for j0,prec_noise_amp in enumerate(new_noise_levels):
            for j1 in range(n_repeats):
                dataset.apply_noise(prec_noise_amp)
                mi[j0,j1] = - model(dataset.X, dataset.Y).detach().cpu().numpy()
        return new_noise_levels, mi