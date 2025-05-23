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
    Maintains only one copy of data in memory, supposedly
    """
    def __init__(self, X, Y, batch_size):
        """
        Args:
            X (torch.Tensor): First time series data of shape [M_x, N]
            Y (torch.Tensor): Second time series data of shape [M_y, N]
            batch_size (int): Size of each batch
        """
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        # Create batch indices
        self.total_batches = (X.shape[1] + batch_size - 1) // batch_size
        # Pre-compute valid batch indices (those with non-zero X and Y)
        self.batch_indices = []
        for i in range(self.total_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, X.shape[1])
            start_ind = start_idx
            end_ind = end_idx
            if torch.any(X[:, start_ind:end_ind] > 0) and torch.any(Y[:, start_ind:end_ind] > 0):
                self.batch_indices.append((start_ind, end_ind))
    def __len__(self):
        return len(self.batch_indices)
    def __getitem__(self, idx):
        """Return a batch at the specified batch index."""
        indices = self.batch_indices[idx]
        return self.X[:, indices[0]:indices[1]], self.Y[:, indices[0]:indices[1]]
    def get_multibatch(self, idxlist):
        """Return a list of batch indices as one vector"""
        indices = torch.concat([torch.arange(s,e) for s,e in [self.batch_indices[idx] for idx in idxlist]])
        return self.X[:, indices], self.Y[:, indices]


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


# Read neuron and muscle data for one moth, return X and Y with specific binning
def process_spike_data(base_name, bin_size=None, neuron_label_filter=None, binarize=True, sample_rate=30000):
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
    use_dtype = np.bool if binarize else np.uint8
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