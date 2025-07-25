import os
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from itertools import product
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from estimators import estimate_mutual_information
from datatools import *
from trainers import *


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


class mlp(nn.Module):
    def __init__(self, in_dim, params):
        """Create an mlp from the configurations."""
        super(mlp, self).__init__()
        # Initialize the layers list
        seq = []
        # Flattening layer
        seq.append(nn.Flatten())
        # Input layer
        seq.append(nn.Linear(in_dim, params['hidden_dim'], bias=params['use_bias']))
        seq.append(params['activation']())
        # Hidden layers
        for i in range(params['layers']):
            layer = nn.Linear(params['hidden_dim'], params['hidden_dim'])
            seq.append(layer)
            seq.append(params['activation']())
        # Connect all together before the output
        self.base_network = nn.Sequential(*seq)
        # Output layer
        self.out = nn.Linear(params['hidden_dim'], params['embed_dim'])
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Pass through fully connected layers
        x = self.base_network(x)
        # Get output
        out = self.out(x)
        return out


class var_mlp(nn.Module):
    def __init__(self, in_dim, params):
        """Create a variational mlp from the configurations."""
        super(var_mlp, self).__init__()
        # Initialize the layers list
        seq = []
        # Flattening layer
        seq.append(nn.Flatten())
        # Input layer
        seq.append(nn.Linear(in_dim, params['hidden_dim']))
        seq.append(params['activation']())
        # nn.init.xavier_uniform_(seq[0].weight)  # Xavier initialization for input layer
        # Hidden layers
        for _ in range(params['layers']):
            layer = nn.Linear(params['hidden_dim'], params['hidden_dim'])
            # nn.init.xavier_uniform_(layer.weight)  # Xavier initialization for hidden layers
            seq.append(layer)
            seq.append(params['activation']())
        # Connect all together before the output
        self.base_network = nn.Sequential(*seq)
        # Two heads for means and log variances
        self.fc_mu = nn.Linear(params['hidden_dim'], params['embed_dim'])
        self.fc_logvar = nn.Linear(params['hidden_dim'], params['embed_dim'])
        # # Initialize the heads with Xavier initialization
        # nn.init.xavier_uniform_(self.fc_mu.weight)
        # nn.init.xavier_uniform_(self.fc_logvar.weight)
        # Normal distribution for sampling
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        # KL Divergence loss initialized to zero
        self.kl_loss = 0.0
        # Set limits for numerical stability
        self.logvar_min = -20  # Lower bound for logVar
        self.logvar_max = 20   # Upper bound for logVar
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
    def __init__(self, in_channels, out_channels, stride=1, dilation=2, activation=nn.ReLU, use_1d_mode=False):
        super(MultiScaleConvBlock, self).__init__()
        # Convert parameters to tuples if use_1d_mode is True
        if use_1d_mode:
            kernel_size = (1, 3)
            stride = (1, stride)
            padding_b1 = (0, 1)
            padding_b2 = (0, dilation)
            dilation = (1, dilation)
        else:
            kernel_size = 3
            stride = stride
            padding_b1 = 1
            padding_b2 = dilation
            dilation = dilation
        # Calculate channels for each branch (distribute output channels more flexibly)
        # Gives larger branches slightly more channels if not divisible by 2
        base_channels = out_channels // 2
        remainder = out_channels % 2
        # Distribute remainder channels to branches (prioritize faster branches)
        branch1_channels = base_channels + remainder  # 3x3 conv  
        branch2_channels = base_channels + remainder  # 3x3 conv dilated
        # Branch 1: 3x3 conv (standard receptive field)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch1_channels, kernel_size=kernel_size, stride=stride, padding=padding_b1, bias=False),
            nn.BatchNorm2d(branch1_channels),
            activation()
        )
        # Branch 2: 3x3 conv with dilation (larger receptive field)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch2_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding_b2, bias=False),
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
    def __init__(self, params, use_1d_mode=False):
        """
        Create a CNN-MLP hybrid model with batch normalization and strided convolutions.
        Input should be a standard params dict. Dict must contain:
        Args:
            hidden_dim (int): Number of hidden units in fully connected layers.
            embed_dim (int): Dimensionality of the output embedding.
            layers (int): Number of convolutional layers.
            fc_layers (int): Number of fully connected layers after GAP.
            activation (str): Activation function to use (e.g., 'relu', 'tanh').
        """
        super(multi_cnn_mlp, self).__init__()
        
        # Multi-scale convolutional layers with strided convolutions and batch normalization
        conv_seq = []
        in_channels = 1
        out_channels = params['n_filters']
        # Please re-write this to not be insane
        for i in range(params['layers']):
            # Determine stride for this layer
            layer_stride = 1 if i == 0 else params['stride']
            
            if params['branch'] is None:
                conv_seq.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=layer_stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    params['activation']()
                ))
            elif params['branch'] == '1':
                if i == 0:
                    conv_seq.append(MultiScaleConvBlock(in_channels, out_channels, stride=1, activation=params['activation']), use_1d_mode=use_1d_mode)
                else:
                    layer_kernel = 3 if not use_1d_mode else (1,3)
                    conv_seq.append(nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=layer_kernel, stride=layer_stride, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        params['activation']()
                    ))
            elif params['branch'] == 'all':
                conv_seq.append(MultiScaleConvBlock(in_channels, out_channels, stride=layer_stride, activation=params['activation'], use_1d_mode=use_1d_mode))
            elif params['branch'] == 'linDilation':
                conv_seq.append(MultiScaleConvBlock(in_channels, out_channels, stride=layer_stride, dilation=i+1, activation=params['activation'], use_1d_mode=use_1d_mode))
            elif params['branch'] == 'multDilation':
                conv_seq.append(MultiScaleConvBlock(in_channels, out_channels, stride=layer_stride, dilation=2*(i+1), activation=params['activation'], use_1d_mode=use_1d_mode))
            else:
                conv_seq.append(MultiScaleConvBlock(in_channels, out_channels, stride=layer_stride, dilation=2**(i+1), activation=params['activation'], use_1d_mode=use_1d_mode))
            # Double the number of filters in each layer
            in_channels = out_channels
            out_channels *= 2
        self.conv_network = nn.Sequential(*conv_seq)
        # Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layers
        fc_seq = []
        in_features = in_channels  # After GAP, the number of features equals the number of channels
        for _ in range(params['fc_layers']):
            fc_seq.append(nn.Linear(in_features, params['hidden_dim']))
            fc_seq.append(params['activation']())
            in_features = params['hidden_dim']
        self.fc_network = nn.Sequential(*fc_seq)
        # Output layer
        self.out = nn.Linear(in_features, params['embed_dim'])
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

    def critic_fn(self, dataZX, dataZY, batch_size=None, gpu_threshold=30000):
        if self.mode in ["sep", "bi"]:
            current_batch_size = dataZX.shape[0]
            
            # If batch is small enough, do it on GPU normally
            if current_batch_size <= gpu_threshold:
                return torch.matmul(dataZY, dataZX.t())
            # For large batches, move to CPU for the matrix multiplication
            device = dataZX.device  # Remember original device
            dataZX_cpu = dataZX.detach().cpu()
            dataZY_cpu = dataZY.detach().cpu()
            # Perform matrix multiplication on CPU
            result_cpu = torch.matmul(dataZY_cpu, dataZX_cpu.t())
            # Move result back to original device
            result = result_cpu.to(device)
            # Clean up CPU tensors
            del dataZX_cpu, dataZY_cpu, result_cpu
            return result
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



def subsample_MI(dataset, params, split_sizes=np.arange(1,6), X='X', Y='Y'):
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
    # Get [start_time, end_time] pairs for each subsample
    split_times = np.empty((0,2), dtype=int)
    for ss in split_sizes:
        # Each row is start, end pair
        split_indices = np.array([[x[0], x[-1]] for x in np.array_split(np.arange(dataset.n_windows), ss)]) 
        split_indices = (split_indices + np.random.choice(dataset.n_windows)) % dataset.n_windows # Shift a random amount
        split_times = np.concatenate((split_times, dataset.window_times[dataset.valid_windows][split_indices]))
    mi = np.zeros((len(split_times)))
    subsets = np.hstack([np.repeat(x, x) for x in split_sizes]) # run length encoding of split_sizes
    # Loop over subsets
    for i in range(split_times.shape[0]):
        # Train model
        mis_test, train_id, indices = train_model_no_eval(dataset, params, X=X, Y=Y, subset_times=split_times[i,:], return_indices=True)
        mod = retrieve_best_model(mis_test, params, train_id=train_id, remove_all=True)
        dataset.move_data_to_windows(time_offset=0)
        # Run model inference to get MI value
        with torch.no_grad():
            mi[i] = - mod(getattr(dataset, X)[indices,:,:], getattr(dataset, Y)[indices,:,:]).detach().cpu().numpy()
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
    # Get [start_time, end_time] pairs for each subsample
    split_times = np.empty((0,2), dtype=int)
    for ss in split_sizes:
        # Each row is start, end pair
        split_indices = np.array([[x[0], x[-1]] for x in np.array_split(np.arange(dataset.n_windows), ss)]) 
        split_indices = (split_indices + np.random.choice(dataset.n_windows)) % dataset.n_windows # Shift a random amount
        split_times = np.concatenate((split_times, dataset.window_times[dataset.valid_windows][split_indices]))
    mi, embed_dim_vec = [], []
    subsets = np.hstack([np.repeat(x, x) for x in split_sizes]) # run length encoding of split_sizes
    # Loop over subsets
    for subset_idx, embed_dim in product(range(len(subsets)), embed_range):
        this_params = {**params, 'embed_dim': embed_dim}
        # Train model
        mis_test, train_id, indices = train_model_no_eval(dataset, this_params, 
            subset_times=split_times[subset_idx,:], return_indices=True, verbose=False
        )
        with torch.no_grad():
            # Retrieve model, run inference to get MI value
            mod = retrieve_best_model(mis_test, this_params, train_id=train_id, remove_all=True)
            dataset.move_data_to_windows(time_offset=0)
            mi.append(- mod(dataset.X[indices,:,:], dataset.Y[indices,:,:]).detach().cpu().numpy())
            embed_dim_vec.append(embed_dim)
    mi = np.array(mi)
    embed_dim_vec = np.array(embed_dim_vec)
    return subsets, mi, embed_dim_vec


def precision_rounding(precision_levels, dataset, model, X='X', Y='Y', 
        early_stop=False, early_stop_threshold=0.5):
    """
    Run spike timing precision analysis
    This version uses rounding instead of added noise
    Args:
        precision_levels: Range of precision levels to run over, in units of seconds
        dataset: TimeWindowDataset
        model: Trained model to run inference with
    Returns:
        mi: Matrix of mutual information at each noise level
    """
    if dataset.use_phase:
        precision_levels = precision_levels / dataset.window_size
    with torch.no_grad():
        # Will always do zero-noise MI
        mi = np.zeros((len(precision_levels)+1))
        mi[0] = - model(getattr(dataset, X), getattr(dataset, Y)).detach().cpu().numpy()
        for i, prec_level in enumerate(precision_levels):
            dataset.apply_precision(prec_level, X=X)
            mi[i+1] = - model(getattr(dataset, X), getattr(dataset, Y)).detach().cpu().numpy()
            if early_stop and (mi[i+1] < (mi[0] * early_stop_threshold)):
                break
        return mi

def precision(noise_levels, dataset, model, n_repeats=3, X='X', Y='Y',
        early_stop=False, early_stop_threshold=0.5):
    """
    Run spike timing precision analysis, to get precision curve
    Args:
        noise_levels: Range of noise levels to run over, in units of seconds
        dataset: BatchedDatasetWithNoise of X and Y data
        model: Trained model to run inference with
        n_repeats: How many times per noise level to repeat
    Returns:
        mi: Matrix of mutual information at each noise level. Rows are repeats, columns noise levels
    """
    if dataset.use_phase:
        noise_levels = noise_levels / dataset.window_size
    with torch.no_grad():
        # Will always do zero-noise MI
        mi = np.zeros((len(noise_levels)+1, n_repeats))
        mi[0,:] = - model(getattr(dataset, X), getattr(dataset, Y)).detach().cpu().numpy()
        for i,prec_noise_amp in enumerate(noise_levels):
            for j in range(n_repeats):
                dataset.apply_noise(prec_noise_amp, X=X)
                mi[i+1,j] = - model(getattr(dataset, X), getattr(dataset, Y)).detach().cpu().numpy()
            if early_stop and (np.mean(mi[i+1,:]) < (mi[0,0] * early_stop_threshold)):
                break
        return mi
