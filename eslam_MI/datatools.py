import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


# Check if CUDA or MPS is running
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = "cpu"



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
        self.Xmain = X
        self.Ymain = Y
        self.X = torch.zeros((self.n_batches, 1, X.shape[0], batch_size), device=device)
        self.Y = torch.zeros((self.n_batches, 1, Y.shape[0], batch_size), device=device)
        for i in range(self.n_batches):
            indices = self.batch_indices[i]
            self.X[i,0,:,:] = X[:, indices[0]:indices[1]]
            self.Y[i,0,:,:] = Y[:, indices[0]:indices[1]]
        # Get spike indices
        self.spike_indices = torch.where(self.X)
        self.spike_indices_Y = torch.where(self.Y)
        # Intermediate tensor for holding noise indices
        self.new_indices = torch.zeros_like(self.spike_indices[3], device=device, dtype=int)
    def __len__(self):
        return self.n_batches
    def __getitem__(self, idx):
        """Return a batch at the specified batch index."""
        return self.X[idx,:,:,:], self.Y[idx,:,:,:]
    def apply_noise(self, amplitude):
        """
        Apply noise to spike times of noisy version of X. 
        Args: 
            amplitude: added uniform noise amplitude, units of samples
        """
        self.X.zero_()
        self.new_indices = torch.clip(
            self.spike_indices[3] + torch.rand(self.spike_indices[3].shape, device=device).mul_(amplitude).round_().int(), 
            0, 
            self.X.shape[3] - 1)
        self.X[self.spike_indices[0], self.spike_indices[1], self.spike_indices[2], self.new_indices] = 1
    def apply_noise_Y(self, amplitude):
        """
        Apply noise to spike times of noisy version of Y. 
        Amplitude is in units of samples
        """
        self.Y.zero_()
        self.new_indices = torch.clip(
            self.spike_indices_Y[3] + torch.rand(self.spike_indices_Y[3].shape, device=device).mul_(amplitude).round_().int(), 
            0, 
            self.Y.shape[3] - 1)
        self.Y[self.spike_indices_Y[0], self.spike_indices_Y[1], self.spike_indices_Y[2], self.new_indices] = 1
    def time_lag(self, lag, channels=None):
        """
        Apply time lag to spike times of all (or specific) neurons/muscles 
        Positive lag shifts entries rightward (forward in time), negative the opposite
        """
        if channels is None:
            channels = torch.arange(self.Xmain.shape[0])
        # Re-make X from rolled copy of Xmain
        tempX = self.Xmain.detach().clone()
        tempX[channels,:] = torch.roll(tempX[channels,:], lag)
        for i in range(self.n_batches):
            indices = self.batch_indices[i]
            self.X[i,0,:,:] = tempX[:, indices[0]:indices[1]]
        del tempX


def create_data_split(dataset, batch_size, train_fraction=0.95, eval_fraction=None, subset_indices=None):
    """
    Creates train loader and test/eval data views
    Args:
        dataset (FullAndBatchedDataset): The dataset containing all data
        train_fraction (float): Fraction of batches to use for training
    Returns:
        tuple: (train_loader, test_indices, eval_indices)
    """
    num_windows = dataset.n_batches if subset_indices is None else len(subset_indices)
    # Generate train/test splits
    train_size = int(train_fraction * num_windows)
    # Create train/test/eval indices, separate eval set with different random indices
    if subset_indices is None:
        traintest_indices = torch.randperm(dataset.n_batches)
    else:
        traintest_indices = subset_indices[torch.randperm(len(subset_indices))]
    train_indices = traintest_indices[:train_size]
    test_indices = traintest_indices[train_size:]
    # Generate eval split, either from train subset or independently
    eval_fraction = (1 - train_fraction) if eval_fraction is None else eval_fraction
    eval_size = int(eval_fraction * num_windows)
    eval_indices = traintest_indices[:eval_size]
    # Create training data loader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=0  # Adjust based on your system
    )
    return train_loader, test_indices, eval_indices


# Read neuron and muscle data for one moth, return X and Y with specific binning
def read_spike_data(base_name, bin_size=None, neuron_label_filter=None, sample_rate=30000, set_precision=0):
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
        set_precision (float): Resolution in s to set precision of data to. Skips this if set_precision=0
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
    X = torch.zeros((len(neuron_labels), max_spike_index + 1), dtype=use_dtype, device=device)  # Neural activity
    Y = torch.zeros((len(muscle_labels), max_spike_index + 1), dtype=use_dtype, device=device)  # Muscle activity
    # Create binary spike trains for neurons
    for i, unit in enumerate(neuron_labels):
        if bin_size is None:
            indices = data[unit]
        else:
            indices = np.rint(data[unit] / scale)
        # Set to precision level if requested
        if set_precision != 0:
            prec_samples = set_precision / bin_size
            indices = np.rint(np.rint(indices / prec_samples) * prec_samples)
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



def save_dicts_to_h5(dicts, filename):
    with h5py.File(filename, 'w') as f:
        for i, d in enumerate(dicts):
            group = f.create_group(f'dict_{i}')
            
            for key, value in d.items():
                if isinstance(value, np.ndarray):
                    group.create_dataset(key, data=value)
                else:
                    # Store strings as fixed-length or variable-length
                    group.create_dataset(key, data=value)

def load_dicts_from_h5(filename):
    dicts = []
    with h5py.File(filename, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            d = {}
            for key in group.keys():
                dataset = group[key]
                if dataset.dtype.char == 'S':  # String data
                    d[key] = dataset[()].decode('utf-8') if isinstance(dataset[()], bytes) else dataset[()]
                else:  # Numeric data
                    d[key] = dataset[()]
            dicts.append(d)
    return dicts