import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
        activation_fn = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            'silu': nn.SiLU,
        }[activation]
    
        # Initialize the layers list
        seq = []
    
        # Input layer
        seq.append(nn.Linear(dim, hidden_dim))
        seq.append(activation_fn())
        nn.init.xavier_uniform_(seq[0].weight)  # Xavier initialization for input layer
    
        # Hidden layers
        for _ in range(layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            nn.init.xavier_uniform_(layer.weight)  # Xavier initialization for hidden layers
            seq.append(layer)
            seq.append(activation_fn())
    
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
        activation_fn = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            'silu': nn.SiLU,
        }[activation]
    
        # Initialize the layers list
        seq = []
    
        # Input layer
        seq.append(nn.Linear(dim, hidden_dim))
        seq.append(activation_fn())
        nn.init.xavier_uniform_(seq[0].weight)  # Xavier initialization for input layer
    
        # Hidden layers
        for _ in range(layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            nn.init.xavier_uniform_(layer.weight)  # Xavier initialization for hidden layers
            seq.append(layer)
            seq.append(activation_fn())
    
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
        return estimate_mutual_information(self.estimator, dataZX, dataZY,
                                        lambda x, y: self.critic_fn(x, y, batch_size),
                                        baseline_fn=self.baseline_fn)

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
        self.all_indices = torch.arange(X.shape[1])
        self.total_columns = X.shape[1]
        self.total_batches = (self.total_columns + batch_size - 1) // batch_size
        # Pre-compute valid batch indices (those with non-zero X and Y)
        self.batch_indices = []
        for i in range(self.total_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, self.total_columns)
            batch_inds = self.all_indices[start_idx:end_idx]
            if torch.any(X[:, batch_inds] > 0) and torch.any(Y[:, batch_inds] > 0):
                self.batch_indices.append(batch_inds)
    def __len__(self):
        return len(self.batch_indices)
    def __getitem__(self, idx):
        """Return a batch at the specified batch index."""
        indices = self.batch_indices[idx]
        return self.X[:, indices], self.Y[:, indices]
    def get_multibatch(self, idxlist):
        """Return a list of batch indices as one vector"""
        indices = torch.concat([self.batch_indices[idx] for idx in idxlist])
        return self.X[:, indices], self.Y[:, indices]

def create_train_test_eval(dataset, train_fraction=0.95, eval_fraction=None, device=None):
    """
    Creates train loader and test/eval data views without concatenation.
    
    Args:
        dataset (FullAndBatchedDataset): The dataset containing all data
        train_fraction (float): Fraction of batches to use for training
        device (torch.device): Device to move test/eval data to
        
    Returns:
        tuple: (train_loader, test_data, eval_data)
    """
    # Generate train/test and eval splits
    train_size = int(train_fraction * len(dataset.batch_indices))
    if eval_fraction is None:
        eval_size = int((1 - train_fraction) * len(dataset.batch_indices))
    else:
        eval_size = int(eval_fraction * len(dataset.batch_indices))
    # Create train/test/eval indices, separate eval set with different random indices
    traintest_indices = torch.randperm(len(dataset.batch_indices))
    train_indices = traintest_indices[:train_size]
    test_indices = traintest_indices[train_size:]
    eval_indices = torch.randperm(len(dataset.batch_indices))[eval_size:]
    # Create training data loader, send test and eval to device
    train_loader = DataLoader(
        dataset,
        sampler=train_indices,
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
def process_spike_data(base_name, bin_size=None, neuron_label_filter=None, sample_rate=30000):
    """
    Processes spike data from a .npz file into neural (X) and muscle (Y) activity tensors.
    
    Parameters:
        base_name (str): Base name of the files (e.g., '2025-03-21').
            The function will look for '_data.npz' and '_labels.npz'.
        bin_size (float, optional): Bin size in seconds. If None, returns raw spikes.
        neuron_label_filter (str, optional): Filter neurons by label (e.g., "good").
            Only used if a labels file is present.
        sample_rate (float, 30000 Hz by default): The sampling rate for the data
        
    Returns:
        X (numpy.ndarray): Neural activity tensor of shape (num_neurons, num_time_points).
        Y (numpy.ndarray): Muscle activity tensor of shape (num_muscles, num_time_points).
        neuron_labels (list): List of neuron labels corresponding to rows in X.
        muscle_labels (list): List of muscle labels corresponding to rows in Y.
    """
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
        X = np.zeros((len(neuron_labels), num_time_points))  # Neural activity
        Y = np.zeros((len(muscle_labels), num_time_points))  # Muscle activity
        # Create binary spike trains for neurons
        for i, unit in enumerate(neuron_labels):
            spike_indices = data[unit]
            X[i, spike_indices] = 1  # Mark spikes as 1 at their respective indices
        # Create binary spike trains for muscles
        for i, unit in enumerate(muscle_labels):
            spike_indices = data[unit]
            Y[i, spike_indices] = 1  # Mark spikes as 1 at their respective indices
    else:
        # Binned spike representation
        bin_samples = int(bin_size * sample_rate)  # Convert bin size to samples
        num_bins = int(np.ceil(max_spike_index / bin_samples))  # Total number of bins
        # Initialize tensors for neural and muscle activity
        X = np.zeros((len(neuron_labels), num_bins))  # Neural activity
        Y = np.zeros((len(muscle_labels), num_bins))  # Muscle activity
        # Bin neural activity
        for i, unit in enumerate(neuron_labels):
            spike_indices = data[unit]
            X[i] = np.histogram(spike_indices, bins=num_bins, range=(0, max_spike_index))[0]
        # Bin muscle activity
        for i, unit in enumerate(muscle_labels):
            spike_indices = data[unit]
            Y[i] = np.histogram(spike_indices, bins=num_bins, range=(0, max_spike_index))[0]
    # Return results
    return X, Y, neuron_labels, muscle_labels