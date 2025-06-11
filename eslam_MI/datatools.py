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
    def __init__(self, X, Y, bout_starts, batch_size, check_activity=False):
        """
        Args:
            X (torch.Tensor): First time series data of shape [M_x, N]
            Y (torch.Tensor): Second time series data of shape [M_y, N]
            batch_size (int): Size of each window
        """
        self.batch_size = batch_size
        # Pre-compute valid batch indices (those with non-zero X and Y)
        self.batch_indices = []
        bout_starts = np.hstack((bout_starts, X.shape[1]))
        bout_diffs = np.diff(bout_starts)
        # Loop over bouts
        for i in range(len(bout_starts)-1):
            # Cut up bout into chunks
            n_chunks_in_bout = np.ceil(bout_diffs[i] / batch_size).astype(int)
            for j in range(n_chunks_in_bout-1):
                start_idx = bout_starts[i] + j * batch_size
                end_idx = start_idx + batch_size
                # If asked, check if activity in X and Y for this window. Can be slow
                if check_activity:
                    should_append = torch.any(X[:, start_idx:end_idx] > 0) and torch.any(Y[:, start_idx:end_idx] > 0)
                    if not should_append:
                        continue
                self.batch_indices.append((start_idx, end_idx))
            # Handle last chunk differently, just go to end of bout
            start_idx = bout_starts[i] + n_chunks_in_bout * batch_size
            end_idx = bout_starts[i+1]
        self.n_batches = len(self.batch_indices)
        # Store X, Y in pre-chunked form
        self.X = torch.zeros((self.n_batches, 1, X.shape[0], batch_size), device=device)
        self.Y = torch.zeros((self.n_batches, 1, Y.shape[0], batch_size), device=device)
        for i in range(self.n_batches):
            # Indexing done this way to catch times when a window is shorter than batch_size
            # In that case we just fill chunk in, leave the rest as zeros
            indices = self.batch_indices[i]
            ind_diff = indices[1] - indices[0]
            self.X[i,0,:,0:ind_diff] = X[:, indices[0]:indices[1]]
            self.Y[i,0,:,0:ind_diff] = Y[:, indices[0]:indices[1]]
    def __len__(self):
        return self.n_batches
    def __getitem__(self, idx):
        """Return a batch at the specified batch index."""
        return self.X[idx,:,:,:], self.Y[idx,:,:,:]


def intersect_sorted_tensors(tensor1, tensor2):
    """
    Efficiently find intersection of two sorted tensors and return
    the indices to filter both tensors to keep only shared elements.
    
    Args:
        tensor1: First sorted tensor
        tensor2: Second sorted tensor
    Returns:
        tuple: (indices1, indices2) where:
            - indices1: indices to filter tensor1
            - indices2: indices to filter tensor2
    """
    # Find where each element of tensor1 would be inserted in tensor2
    indices = torch.searchsorted(tensor2, tensor1)
    # Clamp indices to valid range
    indices = torch.clamp(indices, 0, len(tensor2) - 1)
    # Check which elements actually match
    mask1 = tensor2[indices] == tensor1
    indices1 = torch.where(mask1)[0]
    # For tensor2, we need to find corresponding indices
    # Use searchsorted in reverse
    indices_rev = torch.searchsorted(tensor1, tensor2)
    indices_rev = torch.clamp(indices_rev, 0, len(tensor1) - 1)
    mask2 = tensor1[indices_rev] == tensor2
    indices2 = torch.where(mask2)[0]
    return indices1, indices2



class BatchedDatasetWithNoise(Dataset):
    """
    Dataset that supports both batch-wise and full data access.
    Maintains master copy of data in memory, as well as duplicate on which noise can be applied to spike timings
    Will always return Xnoise, to protect master copy X. 
    X will also always be preserved in [nchannels x ntimepoints] shape, whereas Xnoise is set to chunked/windowed form
    """
    def __init__(self, X, Y, bout_starts, batch_size, check_activity=False):
        """
        Args:
            X (torch.Tensor): First time series data of shape [M_x, N]
            Y (torch.Tensor): Second time series data of shape [M_y, N]
            batch_size (int): Size of each window
        """
        self.batch_size = batch_size
        # Pre-compute valid batch indices (those with non-zero X and Y)
        self.batch_indices = []
        bout_starts = np.hstack((bout_starts, X.shape[1]))
        bout_diffs = np.diff(bout_starts)
        # Loop over bouts
        for i in range(len(bout_starts)-1):
            # Cut up bout into chunks
            n_chunks_in_bout = np.ceil(bout_diffs[i] / batch_size).astype(int)
            for j in range(n_chunks_in_bout-1):
                start_idx = bout_starts[i] + j * batch_size
                end_idx = start_idx + batch_size
                self.batch_indices.append((start_idx, end_idx))
            # Handle last chunk differently, just go to end of bout
            start_idx = bout_starts[i] + n_chunks_in_bout * batch_size
            end_idx = bout_starts[i+1]
        self.n_batches = len(self.batch_indices)
        # Store X, Y in un-chunked form, noise-equivalent versions in pre-chunked form
        self.Xmain = X
        self.Ymain = Y
        self.X = torch.zeros((self.n_batches, 1, X.shape[0], batch_size), device=device)
        self.Y = torch.zeros((self.n_batches, 1, Y.shape[0], batch_size), device=device)
        for i in range(self.n_batches):
            # Indexing done this way to catch times when a window is shorter than batch_size
            # In that case we just fill chunk in, leave the rest as zeros
            indices = self.batch_indices[i]
            ind_diff = indices[1] - indices[0]
            self.X[i,0,:,0:ind_diff] = self.Xmain[:, indices[0]:indices[1]]
            self.Y[i,0,:,0:ind_diff] = self.Ymain[:, indices[0]:indices[1]]
        # Get spike indices
        self.spike_indices = torch.where(self.X)
        self.spike_indices_Y = torch.where(self.Y)
        # If checking activity, remove chunks where there is no activity
        if check_activity:
            # Find batches with activity in both
            keep_batches, validX, validY = np.intersect1d(self.spike_indices[0].cpu(), self.spike_indices_Y[0].cpu(), return_indices=True)
            # Update everything
            self.X = self.X[keep_batches,:,:,:]
            self.Y = self.Y[keep_batches,:,:,:]
            self.n_batches = len(keep_batches)
            self.spike_indices = tuple(tensor[validX] for tensor in self.spike_indices)
            self.spike_indices_Y = tuple(tensor[validY] for tensor in self.spike_indices_Y)
            self.batch_indices = [self.batch_indices[i] for i in keep_batches]
        # Intermediate tensor for holding noise indices
        self.new_indices = torch.zeros_like(self.spike_indices[3], dtype=int, device=device)
    def __len__(self):
        return self.n_batches
    def __getitem__(self, idx):
        """Return a batch at the specified batch index."""
        return self.X[idx,:,:,:], self.Y[idx,:,:,:]
    def apply_noise(self, amplitude):
        """
        Apply noise to spike times of noisy version of X. 
        NOTE: This could become a problem if too many spikes are supposed to move across window borders to next window
        Implementing that was a bit too complicated, and likely doesn't matter. But it might!
        Args: 
            amplitude: added uniform noise amplitude, units of samples
        """
        self.X.zero_()
        self.new_indices = torch.clip(
            self.spike_indices[3] + torch.rand(self.spike_indices[3].shape, device=device).mul_(amplitude).round_().int(), 
            0, self.X.shape[3] - 1)
        self.X[self.spike_indices[0], self.spike_indices[1], self.spike_indices[2], self.new_indices] = 1
    def apply_noise_Y(self, amplitude):
        """
        Apply noise to spike times of noisy version of Y. 
        Amplitude is in units of samples
        """
        self.Y.zero_()
        self.new_indices = torch.clip(
            self.spike_indices_Y[3] + torch.rand(self.spike_indices_Y[3].shape, device=device).mul_(amplitude).round_().int(), 
            0, self.Y.shape[3] - 1)
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
            # Indexing done this way to catch times when a window is shorter than batch_size
            # In that case we just fill chunk in, leave the rest as zeros
            indices = self.batch_indices[i]
            ind_diff = indices[1] - indices[0]
            self.X[i,0,:,0:ind_diff] = tempX[:, indices[0]:indices[1]]
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
def read_spike_data(base_name,
    bin_size=None, neuron_label_filter=None, sample_rate=30000, 
    set_precision_x=0, set_precision_y=0):
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
            bin_size = 1/sample_rate
        else:
            indices = np.rint(data[unit] / scale)
        # Set to precision level if requested
        if set_precision_x != 0:
            prec_samples = set_precision_x / bin_size
            indices = np.clip(np.rint(np.rint(indices / prec_samples) * prec_samples), 0, X.shape[1] - 1)
        X[i, indices] = 1
    # Create binary spike trains for muscles
    for i, unit in enumerate(muscle_labels):
        if bin_size is None:
            indices = data[unit]
            bin_size = 1/sample_rate
        else:
            indices = np.rint(data[unit] / scale)
        if set_precision_y != 0:
            prec_samples = set_precision_y / bin_size
            indices = np.clip(np.rint(np.rint(indices / prec_samples) * prec_samples), 0, Y.shape[1] - 1)
        Y[i, indices] = 1
    # Trim to only bouts
    indices = torch.concat([torch.arange(s,e, dtype=int) for s,e in zip(starts, ends)])
    X = X[:,indices]
    Y = Y[:,indices]
    # Change bout indices to match new length of X and Y
    subtract_amts = np.cumsum(starts - np.hstack((0, ends[0:-1])))
    starts = starts - subtract_amts
    # Close all files
    data.close()
    bouts.close()
    labels_data.close()
    # Return results
    return X, Y, neuron_labels, muscle_labels, starts



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