import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from estimators import estimate_mutual_information
import os
import json


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
    def __init__(self, input_channels, hidden_dim, output_dim, conv_layers, fc_layers, activation):
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
            conv_seq.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if i > 0 else 1, padding=1))
            conv_seq.append(nn.BatchNorm2d(out_channels))  # Add batch normalization
            conv_seq.append(activation())
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

def log_prob_gaussian(x):
    return torch.sum(torch.distributions.Normal(0., 1.).log_prob(x), -1)


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


class BatchedDataset(Dataset):
    """
    Dataset that supports both batch-wise and full data access.
    Maintains only one copy of data in memory
    Simpler, lighter weight version than BatchedDatasetWithNoise
    """
    def __init__(self, X, Y, batch_size, check_activity=False):
        """
        Args:
            X (torch.Tensor): First time series data of shape [M_x, N]
            Y (torch.Tensor): Second time series data of shape [M_y, N]
            batch_size (int): Size of each window
        """
        self.batch_size = batch_size
        # Create batch indices
        self.total_batches = (X.shape[1] + batch_size - 1) // batch_size
        # Pre-compute valid batch indices (those with non-zero X and Y)
        self.batch_indices = []
        for i in range(self.total_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            # Windows must all be same size, ignore last one if too small
            if end_idx > X.shape[1]:
                continue
            # If asked, check if activity in X and Y for this window. Can be slow
            if check_activity:
                should_append = torch.any(X[:, start_idx:end_idx] > 0) and torch.any(Y[:, start_idx:end_idx] > 0)
                if not should_append:
                    continue
            self.batch_indices.append((start_idx, end_idx))
        self.n_batches = len(self.batch_indices)
        # Store X, Y in pre-chunked form
        self.X = torch.zeros((self.n_batches, 1, X.shape[0], batch_size), device=device)
        self.Y = torch.zeros((self.n_batches, 1, Y.shape[0], batch_size), device=device)
        for i in range(self.n_batches):
            indices = self.batch_indices[i]
            self.X[i,0,:,:] = X[:, indices[0]:indices[1]]
            self.Y[i,0,:,:] = Y[:, indices[0]:indices[1]]
    def __len__(self):
        return self.n_batches
    def __getitem__(self, idx):
        """Return a batch at the specified batch index."""
        return self.X[idx,:,:,:], self.Y[idx,:,:,:]


class BatchedDatasetWithNoise(Dataset):
    """
    Dataset that supports both batch-wise and full data access.
    Maintains master copy of data in memory, as well as duplicate on which noise can be applied to spike timings
    Will always return Xnoise, to protect master copy X. 
    X will also always be preserved in [nchannels x ntimepoints] shape, whereas Xnoise is set to chunked/windowed form
    """
    def __init__(self, X, Y, batch_size, check_activity=False):
        """
        Args:
            X (torch.Tensor): First time series data of shape [M_x, N]
            Y (torch.Tensor): Second time series data of shape [M_y, N]
            batch_size (int): Size of each window
        """
        self.batch_size = batch_size
        # Create batch indices
        self.total_batches = (X.shape[1] + batch_size - 1) // batch_size
        # Pre-compute valid batch indices (those with non-zero X and Y)
        self.batch_indices = []
        for i in range(self.total_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            # Windows must all be same size, ignore last one if too small
            if end_idx > X.shape[1]:
                continue
            # If asked, check if activity in X and Y for this window. Can be slow
            if check_activity:
                should_append = torch.any(X[:, start_idx:end_idx] > 0) and torch.any(Y[:, start_idx:end_idx] > 0)
                if not should_append:
                    continue
            self.batch_indices.append((start_idx, end_idx))
        self.n_batches = len(self.batch_indices)
        # Store X, Y in un-chunked form, noise-equivalent versions in pre-chunked form
        self.X = X
        self.Y = Y
        self.Xnoise = torch.zeros((self.n_batches, 1, X.shape[0], batch_size), device=device)
        self.Ynoise = torch.zeros((self.n_batches, 1, Y.shape[0], batch_size), device=device)
        for i in range(self.n_batches):
            indices = self.batch_indices[i]
            self.Xnoise[i,0,:,:] = X[:, indices[0]:indices[1]]
            self.Ynoise[i,0,:,:] = Y[:, indices[0]:indices[1]]
        # Get spike indices
        self.spike_indices = torch.where(self.Xnoise)
        self.spike_indices_Y = torch.where(self.Ynoise)
    def __len__(self):
        return self.n_batches
    def __getitem__(self, idx):
        """Return a batch at the specified batch index."""
        return self.Xnoise[idx,:,:,:], self.Ynoise[idx,:,:,:]
    def apply_noise(self, amplitude):
        """
        Apply noise to spike times of noisy version of X. 
        Amplitude is in units of samples
        """
        self.Xnoise.zero_()
        noise = torch.rand(self.spike_indices[3].shape, device=device)
        noise.mul_(amplitude).round_()  # In-place operations
        new_indices = torch.clip(self.spike_indices[3] + noise.int(), 0, self.Xnoise.shape[3] - 1)
        self.Xnoise[self.spike_indices[0], self.spike_indices[1], self.spike_indices[2], new_indices] = 1
    def apply_noise_Y(self, amplitude):
        """
        Apply noise to spike times of noisy version of Y. 
        Amplitude is in units of samples
        """
        self.Ynoise.zero_()
        noise = torch.rand(self.spike_indices[3].shape, device=device)
        noise.mul_(amplitude).round_()  # In-place operations
        new_indices = torch.clip(self.spike_indices[3] + noise.int(), 0, self.Ynoise.shape[3] - 1)
        self.Ynoise[self.spike_indices[0], self.spike_indices[1], self.spike_indices[2], new_indices] = 1
    def time_lag(self, lag, channels=None):
        """
        Apply time lag to spike times of all (or specific) neurons/muscles 
        Positive lag shifts entries rightward (forward in time), negative the opposite
        """
        if channels is None:
            channels = torch.arange(self.X.shape[0])
        # Re-make Xnoise from rolled copy of X
        tempX = self.X.detach().clone()
        tempX[channels,:] = torch.roll(tempX[channels,:], lag)
        for i in range(self.n_batches):
            indices = self.batch_indices[i]
            self.Xnoise[i,0,:,:] = tempX[:, indices[0]:indices[1]]


def create_data_split(dataset, train_fraction=0.95, eval_fraction=None, eval_from_train=True, device=None):
    """
    Creates train loader and test/eval data views
    Args:
        dataset (FullAndBatchedDataset): The dataset containing all data
        train_fraction (float): Fraction of batches to use for training
        device (torch.device): Device to move test/eval data to
        
    Returns:
        tuple: (train_loader, test_data, eval_data)
    """
    # Generate train/test splits
    train_size = int(train_fraction * len(dataset.batch_indices))
    # Create train/test/eval indices, separate eval set with different random indices
    traintest_indices = torch.randperm(len(dataset.batch_indices))
    train_indices = traintest_indices[:train_size]
    test_indices = traintest_indices[train_size:]
    # Generate eval split, either from train subset or independently
    eval_fraction = (1 - train_fraction) if eval_fraction is None else eval_fraction
    eval_size = int(eval_fraction * len(dataset.batch_indices))
    if eval_from_train:
        eval_indices = traintest_indices[:eval_size]
    else:
        eval_indices = torch.randperm(len(dataset.batch_indices))[:eval_size]
    # Create training data loader, send test and eval to device
    train_loader = DataLoader(
        dataset,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=0  # Adjust based on your system
    )
    # Get test and eval data as views. Batches are sorted to be presented in real data order
    test_X, test_Y = dataset.get_multibatch(test_indices.sort()[0])
    eval_X, eval_Y = dataset.get_multibatch(eval_indices.sort()[0])
    # Transpose test and eval data
    test_X, test_Y, eval_X, eval_Y = test_X.T, test_Y.T, eval_X.T, eval_Y.T
    # Move test and eval data to device if specified
    if device is not None:
        test_X = test_X.to(device)
        test_Y = test_Y.to(device)
        eval_X = eval_X.to(device)
        eval_Y = eval_Y.to(device)
    return train_loader, (test_X, test_Y), (eval_X, eval_Y)

def create_cnn_data_split(dataset, batch_size, train_fraction=0.95, eval_fraction=None, eval_from_train=True):
    """
    Creates train loader and test/eval data views
    Args:
        dataset (FullAndBatchedDataset): The dataset containing all data
        train_fraction (float): Fraction of batches to use for training
    Returns:
        tuple: (train_loader, test_indices, eval_indices)
    """
    # Generate train/test splits
    train_size = int(train_fraction * len(dataset.batch_indices))
    # Create train/test/eval indices, separate eval set with different random indices
    traintest_indices = torch.randperm(len(dataset.batch_indices))
    train_indices = traintest_indices[:train_size]
    test_indices = traintest_indices[train_size:]
    # Generate eval split, either from train subset or independently
    eval_fraction = (1 - train_fraction) if eval_fraction is None else eval_fraction
    eval_size = int(eval_fraction * len(dataset.batch_indices))
    if eval_from_train:
        eval_indices = traintest_indices[:eval_size]
    else:
        eval_indices = torch.randperm(len(dataset.batch_indices))[:eval_size]
    # Create training data loader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=0  # Adjust based on your system
    )
    return train_loader, test_indices, eval_indices


# HISTORICAL FUNCTION. Kept in case of future need. Use read_spike_data, much faster (though only binarizes)
def process_spike_data_old(base_name, bin_size=None, neuron_label_filter=None, binarize=True, sample_rate=30000):
    """
    Processes spike data from a .npz file into neural (X) and muscle (Y) activity tensors.
    Returns either uint8 or bool (if binarized)
    
    Parameters:
        base_name (str): Base name of the files (e.g., '2025-03-21').
            The function will look for '_data.npz' and '_labels.npz'.
        bin_size (float, optional): Bin size in seconds. If None, returns raw spikes.
        neuron_label_filter (str, optional): Filter neurons by label (e.g., "good").
            Only used if a labels file is present.
        sample_rate (float, 30000 Hz by default): The sampling rate for the data
        binarize (bool): Whether to just return boolean of whether ANY spikes were in bin, rather than number
        
    Returns:
        X (numpy.ndarray): Neural activity tensor of shape (num_neurons, num_time_points).
        Y (numpy.ndarray): Muscle activity tensor of shape (num_muscles, num_time_points).
        neuron_labels (list): List of neuron labels corresponding to rows in X.
        muscle_labels (list): List of muscle labels corresponding to rows in Y.
    """
    # Type handling
    use_dtype = bool if binarize else np.uint8
    # Construct file paths
    data_file = f"{base_name}_data.npz"
    labels_file = f"{base_name}_labels.npz"
    # Load the spike data
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file '{data_file}' not found.")
    data = np.load(data_file)
    units = data.files  # List of unit labels
    # Load the labels file if it exists
    labels = {}
    if os.path.exists(labels_file):
        labels_data = np.load(labels_file)
        labels = {unit: labels_data[unit].item() for unit in labels_data.files}
    # Separate neurons and muscles, applying filtering if needed
    neuron_labels = []
    muscle_labels = []
    for unit in units:
        if unit.isnumeric():  # Numeric labels are neurons
            if labels:
                label = labels.get(unit, None)
                if neuron_label_filter is None or label == neuron_label_filter:
                    neuron_labels.append(unit)
            else:
                neuron_labels.append(unit)  # No filtering if no labels file
        else:  # Alphabetic labels are muscles
            muscle_labels.append(unit)
    # Find the maximum spike index to determine total time points
    max_spike_index = max([data[unit].max() for unit in units])
    if bin_size is None:
        # Raw spike representation
        num_time_points = max_spike_index + 1  # Include the last time step
        # Initialize tensors for neural and muscle activity
        X = np.zeros((len(neuron_labels), num_time_points), dtype=use_dtype)  # Neural activity
        Y = np.zeros((len(muscle_labels), num_time_points), dtype=use_dtype)  # Muscle activity
        # Create binary spike trains for neurons
        for i, unit in enumerate(neuron_labels):
            spike_indices = data[unit]
            X[i, spike_indices] = 1  # Mark spikes as 1 at their respective indices
        # Create binary spike trains for muscles
        for i, unit in enumerate(muscle_labels):
            spike_indices = data[unit]
            Y[i, spike_indices] = 1  # Mark spikes as 1 at their respective indices
    else:
        # Convert indices at sample frequency to indices at new size, set to 1
        # Binned spike representation
        bin_samples = int(bin_size * sample_rate)  # Convert bin size to samples
        num_bins = int(np.ceil(max_spike_index / bin_samples))  # Total number of bins
        # Initialize tensors for neural and muscle activity
        X = np.zeros((len(neuron_labels), num_bins), dtype=use_dtype)  # Neural activity
        Y = np.zeros((len(muscle_labels), num_bins), dtype=use_dtype)  # Muscle activity
        # Bin neural activity
        for i, unit in enumerate(neuron_labels):
            spike_indices = data[unit]
            X[i,:] = np.histogram(spike_indices, bins=num_bins, range=(0, max_spike_index))[0]
        # Bin muscle activity
        for i, unit in enumerate(muscle_labels):
            spike_indices = data[unit]
            Y[i,:] = np.histogram(spike_indices, bins=num_bins, range=(0, max_spike_index))[0]
    # Return results
    return X, Y, neuron_labels, muscle_labels


# Read neuron and muscle data for one moth, return X and Y with specific binning
def read_spike_data(base_name, bin_size=None, neuron_label_filter=None, sample_rate=30000):
    """
    Processes spike data from 3 .npz files into neural (X) and muscle (Y) activity tensors.
    Returns bool, always binarizes. If a bin has multiple spikes occur, will just appear as 1
    Requires a data, labels, and bouts file for each moth. 
    Assumes no spike data occurs outside of bouts
    
    Parameters:
        base_name (str): Base name of the files (e.g., '2025-03-21').
            The function will look for '_data.npz' and '_labels.npz'.
        bin_size (float, optional): Bin size in seconds. If None, returns spikes at sample rate
        neuron_label_filter (str, optional): Filter neurons by label (e.g., "good").
            Only used if a labels file is present.
        sample_rate (float, 30000 Hz by default): The sampling rate for the data
        
    Returns:
        X (numpy.ndarray): Neural activity tensor of shape (num_neurons, num_time_points).
        Y (numpy.ndarray): Muscle activity tensor of shape (num_muscles, num_time_points).
        neuron_labels (list): List of neuron labels corresponding to rows in X.
        muscle_labels (list): List of muscle labels corresponding to rows in Y.
    """
    # Type handling
    use_dtype = torch.bool
    # Scale factor if downsampling needed
    scale = bin_size * sample_rate if bin_size is not None else 1
    # Construct file paths
    data_file = f"{base_name}_data.npz"
    labels_file = f"{base_name}_labels.npz"
    bouts_file = f"{base_name}_bouts.npz"
    # Load the spike data
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file '{data_file}' not found.")
    data = np.load(data_file)
    units = data.files  # List of unit labels
    # Load the labels file if it exists
    labels = {}
    if os.path.exists(labels_file):
        labels_data = np.load(labels_file)
        labels = {unit: labels_data[unit].item() for unit in labels_data.files}
    # Load bout start and end indices, rescale by scale factor. (-1 is b/c coming from 1-index Julia)
    bouts = np.load(bouts_file)
    starts, ends = bouts['starts'], bouts['ends']
    starts = np.rint((starts - 1) / scale).astype(int)
    ends = np.rint((ends - 1) / scale).astype(int)
    # Separate neurons and muscles, applying filtering if needed
    neuron_labels = []
    muscle_labels = []
    for unit in units:
        if unit.isnumeric():  # Numeric labels are neurons
            if labels:
                label = labels.get(unit, None)
                if neuron_label_filter is None or label == neuron_label_filter:
                    neuron_labels.append(unit)
            else:
                neuron_labels.append(unit)  # No filtering if no labels file
        else:  # Alphabetic labels are muscles
            muscle_labels.append(unit)
    # Find the maximum spike index to determine total time points
    max_spike_index = ends[-1]
    # Initialize tensors for neural and muscle activity
    X = torch.zeros((len(neuron_labels), max_spike_index + 1), dtype=use_dtype)  # Neural activity
    Y = torch.zeros((len(muscle_labels), max_spike_index + 1), dtype=use_dtype)  # Muscle activity
    # Create binary spike trains for neurons
    for i, unit in enumerate(neuron_labels):
        if bin_size is None:
            indices = data[unit]
        else:
            indices = np.rint(data[unit] / scale)
        X[i, indices] = 1
    # Create binary spike trains for muscles
    for i, unit in enumerate(muscle_labels):
        if bin_size is None:
            indices = data[unit]
        else:
            indices = np.rint(data[unit] / scale)
        Y[i, indices] = 1
    # Trim to only bouts
    indices = torch.concat([torch.arange(s,e, dtype=int) for s,e in zip(starts, ends)])
    X = X[:,indices]
    Y = Y[:,indices]
    # Return results
    return X, Y, neuron_labels, muscle_labels


"""
Function workflow designed for performance, with precision estimation in mind
Involves getting spike indices from data first with get_spike_indices
New datasets are generated from indices instead of vectors
"""
def get_spike_indices(base_name, neuron_label_filter=None):
    # File names
    data_file = base_name + "_data.npz"
    labels_file = base_name + "_labels.npz"
    # Load the spike data
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file '{data_file}' not found.")
    data = np.load(data_file)
    units = data.files  # List of unit labels
    # Load the labels file if it exists
    labels = {}
    if os.path.exists(labels_file):
        labels_data = np.load(labels_file)
        labels = {unit: labels_data[unit].item() for unit in labels_data.files}
    # Separate neurons and muscles, applying filtering if needed
    neuron_labels = []
    muscle_labels = []
    for unit in units:
        if unit.isnumeric():  # Numeric labels are neurons
            if labels:
                label = labels.get(unit, None)
                if neuron_label_filter is None or label == neuron_label_filter:
                    neuron_labels.append(unit)
            else:
                neuron_labels.append(unit)  # No filtering if no labels file
        else:  # Alphabetic labels are muscles
            muscle_labels.append(unit)
    return data, neuron_labels, muscle_labels