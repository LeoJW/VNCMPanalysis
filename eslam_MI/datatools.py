import os
import time
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from itertools import product

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


def max_events_in_window(event_times, window_size):
    """
    Find the maximum number of events that can fit in a window of given size.
    
    Args:
        event_times: Array-like of event timestamps
        window_size: Length of the time window
        
    Returns:
        Maximum number of events that fit in any window of the given size
    """
    times = np.array(event_times)    
    max_count = 0
    left = 0
    for right in range(len(times)):
        # Slide left pointer forward until window contains times[right] - window_size
        while times[right] - times[left] > window_size:
            left += 1
        # Update max_count if current window is larger
        max_count = max(max_count, right - left + 1)
    
    return max_count


def monotonic_repeats_to_ranges(a):
    unique, index, counts = np.unique(a, return_counts=True, return_index=True)
    arg_s = np.argsort(index)
    return np.concatenate(list(map(np.arange,counts[arg_s])), axis=0)


class TimeWindowDataset(Dataset):
    """
    Reads in data, stores and outputs in form similar to previous KSG encoding
    Maintains master copy of data in memory, as well as duplicate on which noise/other things can be applied
    Will always return X, to protect master copy Xmain. 
    TODO: Could do times in milliseconds, would potentially enable a lower precision dtype 
    """
    def __init__(self, base_name, window_size, 
            no_spike_value=0, # Value to use as filler for no spikes (leave at zero)
            time_offset=0.0, # Time offset to apply to everything
            use_ISI=True, # Whether to encode spikes by absolute time in window, or ISI
            sample_rate=30000, # Don't change
            neuron_label_filter=None, # Whether to take good (1), MUA (0), or all (None) neurons
            select_x=None, select_y=None, # Variable selection
        ):
        self.read_data(base_name, sample_rate=sample_rate, neuron_label_filter=neuron_label_filter)
        
        # Could write this so this is adjustable after dataset is made. I'm lazy for now though
        if select_x is not None:
            self.Xtimes = [self.Xtimes[i] for i in select_x]
        if select_y is not None:
            self.Ytimes = [self.Ytimes[i] for i in select_y]
        
        self.use_ISI = use_ISI
        self.window_size = window_size
        self.no_spike_value = no_spike_value
        
        self.move_data_to_windows(time_offset=time_offset)
    
    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, idx):
        """Return a batch at the specified batch index."""
        return self.X[idx,:,:], self.Y[idx,:,:]
    
    def apply_noise(self, amplitude, X='X'):
        """
        Apply noise to spike times of noisy version of whichever variable (X,Y,Z) selected
        Args: 
            amplitude: added uniform noise amplitude, units of seconds
        """
        # Generate noise directly into pre-allocated buffer
        getattr(self, 'noise_buffer_' + X.lower()).uniform_(0, amplitude)
        # Apply noise in-place
        mask = getattr(self, 'spike_mask_' + X.lower())
        getattr(self, X)[mask] = getattr(self, X + 'main')[mask] + getattr(self, 'noise_buffer_' + X.lower())
    
    def apply_precision(self, prec, X='X'):
        """ Set data to a specific precision level prec. Units are same as spike times (s)"""
        mask = getattr(self, 'spike_mask_' + X.lower())
        getattr(self, X)[mask] = torch.round(getattr(self, X + 'main')[mask] / prec) * prec
    
    def move_data_to_windows(self, time_offset=0.0):
        """
        Take list of spike times, convert to matrices of spike times in given windows
        Args:
            no_spike_value: Filler value used to indicate no spike occurred in that position
            time_offset: Offset to window start time in s, should be positive!
        """
        # Make chunk start times
        # Loop over bouts, cut up into chunks, assign start times
        bout_diffs = self.bout_ends - self.bout_starts
        window_times, valid_window = [], []
        # Partial-length windows will often exist at end of bouts, valid_window is used to trim this.
        # If time offset added, this also ends up trimming partial-length window at start of next bout (great!).
        # Just leaves partial-length start of very first bout, if an offset is added
        # For that we add a starting window to catch spikes before initial time offset
        if time_offset > 0:
            window_times.append(0)
            valid_window.append(False)
        for i in range(len(self.bout_starts)):
            n_windows = np.ceil((bout_diffs[i] - time_offset) / self.window_size).astype(int)
            for j in range(n_windows):
                start_time = self.bout_starts[i] + time_offset + j * self.window_size
                window_times.append(start_time)
                valid_window.append(True)
            # If in between bouts, make a window edge that's halfway, this window should always be invalid
            if i < (len(self.bout_starts) - 1):
                start_time = self.bout_ends[i] + (self.bout_starts[i+1] - self.bout_ends[i])
                window_times.append(start_time)
                valid_window.append(False)
            # Last window in any bout is automatically made invalid
            valid_window[-1] = False
        self.valid_window = np.array(valid_window)
        self.window_times = np.array(window_times)
        self.n_windows = len(self.window_times)
        # Use searchsorted on each neuron/muscle to assign to chunks
        # This is by far the slowest part of this whole function!
        window_inds_x = [np.searchsorted(self.window_times, x) - 1 for x in self.Xtimes]
        window_inds_y = [np.searchsorted(self.window_times, y) - 1 for y in self.Ytimes]
        # Find max possible number of spikes per window for X and Y, preallocate. 
        # This MUST only be done once, so that network size is consistent as windows get shifted
        if not hasattr(self, 'max_neuron'):
            self.max_neuron = np.max(np.array([max_events_in_window(x, self.window_size) for x in self.Xtimes]))
            self.max_muscle = np.max(np.array([max_events_in_window(y, self.window_size) for y in self.Ytimes]))
        # Preallocate (size is [window, neuron/muscle, spike time])
        Xmain = np.full((self.n_windows, len(self.Xtimes), self.max_neuron), self.no_spike_value, dtype=np.float32)
        Ymain = np.full((self.n_windows, len(self.Ytimes), self.max_muscle), self.no_spike_value, dtype=np.float32)
        # Loop down each neuron, muscle, assign data to main arrays
        if self.use_ISI:
            for i in range(len(self.Xtimes)):
                column_inds = monotonic_repeats_to_ranges(window_inds_x[i])
                firstval = self.Xtimes[i][0] - self.window_times[window_inds_x[i]][0]
                Xmain[window_inds_x[i],i,column_inds] = np.insert(np.diff(self.Xtimes[i] - self.window_times[window_inds_x[i]]), 0, firstval)
                first = column_inds == 0
                Xmain[window_inds_x[i][first], i, column_inds[first]] = self.Xtimes[i][first] - self.window_times[window_inds_x[i][first]]
            for i in range(len(self.Ytimes)):
                column_inds = monotonic_repeats_to_ranges(window_inds_y[i])
                firstval = self.Ytimes[i][0] - self.window_times[window_inds_y[i]][0]
                Ymain[window_inds_y[i],i,column_inds] = np.insert(np.diff(self.Ytimes[i] - self.window_times[window_inds_y[i]]), 0, firstval)
                first = column_inds == 0
                Ymain[window_inds_y[i][first], i, column_inds[first]] = self.Ytimes[i][first] - self.window_times[window_inds_y[i][first]]
        else:    
            for i in range(len(self.Xtimes)):
                column_inds = monotonic_repeats_to_ranges(window_inds_x[i])
                Xmain[window_inds_x[i],i,column_inds] = self.Xtimes[i] - self.window_times[window_inds_x[i]]
            for i in range(len(self.Ytimes)):
                column_inds = monotonic_repeats_to_ranges(window_inds_y[i])
                Ymain[window_inds_y[i],i,column_inds] = self.Ytimes[i] - self.window_times[window_inds_y[i]]
        # Trim windows that are too short to be valid
        Xmain = Xmain[self.valid_window,:,:]
        Ymain = Ymain[self.valid_window,:,:]
        self.window_times = self.window_times[self.valid_window]
        self.n_windows = len(self.window_times)
        # Convert to tensor, move to device. Make copies that noise will be applied on
        self.Xmain = torch.tensor(Xmain, device=device)
        self.Ymain = torch.tensor(Ymain, device=device)
        self.X = self.Xmain.detach().clone()
        self.Y = self.Ymain.detach().clone()
        # Pre-compute mask of where actual spikes are
        self.spike_mask_x = torch.nonzero(self.Xmain != self.no_spike_value, as_tuple=True)
        self.spike_mask_y = torch.nonzero(self.Ymain != self.no_spike_value, as_tuple=True)
        # Pre-allocate noise tensor, number of spikes to avoid repeated allocations/operations when applying noise
        self.noise_buffer_x = torch.empty(len(self.spike_mask_x[0]), device=device, dtype=self.X.dtype)
        self.noise_buffer_y = torch.empty(len(self.spike_mask_y[0]), device=device, dtype=self.Y.dtype)
    # def time_lag(self, lag, channels=None):
    #     """
    #     Apply time lag to spike times of all (or specific) neurons/muscles 
    #     Positive lag shifts entries rightward (forward in time), negative the opposite
    #     """
    #     if channels is None:
    #         channels = torch.arange(self.Xmain.shape[0])
    #     # Re-make X from rolled copy of Xmain
    #     tempX = self.Xmain.detach().clone()
    #     tempX[channels,:] = torch.roll(tempX[channels,:], lag)
    #     for i in range(self.n_windows):
    #         # Indexing done this way to catch times when a window is shorter than batch_size
    #         # In that case we just fill chunk in, leave the rest as zeros
    #         indices = self.batch_indices[i]
    #         ind_diff = indices[1] - indices[0]
    #         self.X[i,0,:,0:ind_diff] = tempX[:, indices[0]:indices[1]]
    #     del tempX

    def read_data(self, base_name, sample_rate=30000, neuron_label_filter=None):
        """
        Processes spike data from 3 .npz files into neural (X) and muscle (Y) activity tensors.
        """
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
        # Load bout start and end indices, rescale to seconds. (-1 is b/c coming from 1-index Julia)
        bouts = np.load(bouts_file)
        starts, ends = bouts['starts'], bouts['ends']
        self.bout_starts = (starts - 1) / sample_rate
        self.bout_ends = (ends - 1) / sample_rate
        # Separate neurons and muscles, applying filtering if needed
        self.neuron_labels, self.muscle_labels = [], []
        for unit in units:
            if unit.isnumeric():  # Numeric labels are neurons
                if labels:
                    label = labels.get(unit, None)
                    if neuron_label_filter is None or label == neuron_label_filter:
                        self.neuron_labels.append(unit)
                else:
                    self.neuron_labels.append(unit)  # No filtering if no labels file
            else:  # Alphabetic labels are muscles
                self.muscle_labels.append(unit)
        # Make list of neuron (X) and muscle (Y) spike times
        self.Xtimes, self.Ytimes = [], []
        for unit in self.neuron_labels:
            self.Xtimes.append(np.array(data[unit] / sample_rate))
        for unit in self.muscle_labels:
            self.Ytimes.append(np.array(data[unit] / sample_rate))
        # Close all files
        data.close()
        bouts.close()
        labels_data.close()


class TimeWindowDatasetKinematics(Dataset):
    """
    Reads in data, stores and outputs in form similar to previous KSG encoding
    This version also reads in kinematics data, stores it as (Z)
    Maintains master copy of data in memory, as well as duplicate on which noise/other things can be applied
    Will always return X, to protect master copy Xmain. 
    """
    def __init__(self, base_name, window_size, 
            no_spike_value=0, # Value to use as filler for no spikes (leave at zero)
            time_offset=0.0, # Time offset to apply to everything
            use_ISI=True, # Whether to encode spikes by absolute time in window, or ISI
            sample_rate=30000, # Don't change
            neuron_label_filter=None, # Whether to take good (1), MUA (0), or all (None) neurons
            select_x=None, select_y=None, # Variable selection
            angles_only=True, # Only use wing angles (no xyz point data)
            only_valid_kinematics=False, # Only keep windows with kinematics
            only_flapping=False
        ):
        self.read_data(base_name, sample_rate=sample_rate, neuron_label_filter=neuron_label_filter)
        
        # Could write this so this is adjustable after dataset is made. I'm lazy for now though
        # Select from X by index, or by name
        if select_x is not None and all(isinstance(x, (int, float)) for x in select_x):
            self.Xtimes = [self.Xtimes[i] for i in select_x]
            self.neuron_labels = [self.neuron_labels[i] for i in select_x]
        elif select_x is not None:
            keep_inds = [i for i,lab in enumerate(self.neuron_labels) if lab in select_x]
            self.Xtimes = [self.Xtimes[i] for i in keep_inds]
            self.neuron_labels = [self.neuron_labels[i] for i in keep_inds]
        # Select from Y by index, or by name
        if select_y is not None and all(isinstance(x, (int, float)) for x in select_y):
            self.Ytimes = [self.Ytimes[i] for i in select_y]
            self.muscle_labels = [self.muscle_labels[i] for i in select_y]
        elif select_y is not None:
            keep_inds = [i for i,lab in enumerate(self.muscle_labels) if lab in select_y]
            self.Ytimes = [self.Ytimes[i] for i in keep_inds]
            self.muscle_labels = [self.muscle_labels[i] for i in keep_inds]
        if angles_only is not None:
            self.Zorig = self.Zorig[0:6,:]
            self.kine_names = self.kine_names[0:6]
        
        self.use_ISI = use_ISI
        self.window_size = window_size
        self.no_spike_value = no_spike_value
        self.only_valid_kinematics = only_valid_kinematics
        self.only_flapping = only_flapping
        
        self.move_data_to_windows(time_offset=time_offset)
    
    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, idx):
        """Return a batch at the specified batch index."""
        return self.X[idx,:,:], self.Y[idx,:,:]
    
    def apply_noise(self, amplitude, X='X'):
        """
        Apply noise to spike times of noisy version of whichever variable (X,Y,Z) selected
        Args: 
            amplitude: added uniform noise amplitude, units of seconds
        """
        # Generate noise directly into pre-allocated buffer
        getattr(self, 'noise_buffer_' + X.lower()).uniform_(0, amplitude)
        # Apply noise in-place
        mask = getattr(self, 'spike_mask_' + X.lower())
        getattr(self, X)[mask] = getattr(self, X + 'main')[mask] + getattr(self, 'noise_buffer_' + X.lower())
    
    def apply_precision(self, prec, X='X'):
        """ Set data to a specific precision level prec. Units are same as spike times (s)"""
        mask = getattr(self, 'spike_mask_' + X.lower())
        getattr(self, X)[mask] = torch.round(getattr(self, X + 'main')[mask] / prec) * prec
    
    def time_shift(self, time_offset=0.0, X='X'):
        """
        Shift one variable (X, can be 'X', 'Y', or 'Z') in time by time_offset
        Runs basically the same thing as move_data_to_windows, but on one variable, and matches the resulting windows up to 
        windows for the rest of the dataset
        """
        # Do everything as you would

        # Run a searchsorted on new window_times into main window_times. Use this to match 1:1
        # Have some mechanism to mark these matches invalid if too far apart
    
    def move_data_to_windows(self, time_offset=0.0):
        """
        Take list of spike times, convert to matrices of spike times in given windows
        Args:
            no_spike_value: Filler value used to indicate no spike occurred in that position
            time_offset: Offset to window start time in s, should be positive!
        """
        # Make chunk start times
        # Loop over bouts, cut up into chunks, assign start times
        bout_diffs = self.bout_ends - self.bout_starts
        window_times, valid_window = [], []
        # Partial-length windows will often exist at end of bouts, valid_window is used to trim this.
        # If time offset added, this also ends up trimming partial-length window at start of next bout (great!).
        # Just leaves partial-length start of very first bout, if an offset is added
        # For that we add a starting window to catch spikes before initial time offset
        if time_offset > 0:
            window_times.append(0)
            valid_window.append(False)
        for i in range(len(self.bout_starts)):
            n_windows = np.ceil((bout_diffs[i] - time_offset) / self.window_size).astype(int)
            for j in range(n_windows):
                start_time = self.bout_starts[i] + time_offset + j * self.window_size
                window_times.append(start_time)
                valid_window.append(True)
            # If in between bouts, make a window edge that's halfway, this window should always be invalid
            if i < (len(self.bout_starts) - 1):
                start_time = self.bout_ends[i] + (self.bout_starts[i+1] - self.bout_ends[i])
                window_times.append(start_time)
                valid_window.append(False)
            # Last window in any bout is automatically made invalid
            valid_window[-1] = False
        self.valid_window = np.array(valid_window)
        self.window_times = np.array(window_times)
        self.n_windows = len(self.window_times)
        # For kinematics valid windows are only those of the correct length (since it's not an event-based process)
        window_valid_length = np.diff(self.window_times) < (self.window_size + np.finfo(np.float32).eps)
        self.kine_valid_window = np.logical_and(self.valid_window, np.concatenate((window_valid_length, np.array([True]))))
        # Use searchsorted on each neuron/muscle to assign to chunks
        # This is by far the slowest part of this whole function!
        window_inds_x = [np.searchsorted(self.window_times, x) - 1 for x in self.Xtimes]
        window_inds_y = [np.searchsorted(self.window_times, y) - 1 for y in self.Ytimes]
        window_inds_z = np.searchsorted(self.window_times, self.Ztimes, side='right') - 1
        # Find max possible number of spikes per window for X and Y, preallocate. 
        # This MUST only be done once, so that network size is consistent as windows get shifted
        if not hasattr(self, 'max_neuron'):
            self.max_neuron = np.max(np.array([max_events_in_window(x, self.window_size) for x in self.Xtimes]))
            self.max_muscle = np.max(np.array([max_events_in_window(y, self.window_size) for y in self.Ytimes]))
            self.max_kine = np.ceil(self.window_size / np.mean(np.diff(self.Ztimes[:10]))).astype(int)
        # Preallocate (size is [window, neuron/muscle, spike time])
        Xmain = np.full((self.n_windows, len(self.Xtimes), self.max_neuron), self.no_spike_value, dtype=np.float32)
        Ymain = np.full((self.n_windows, len(self.Ytimes), self.max_muscle), self.no_spike_value, dtype=np.float32)
        Zmain = np.full((self.n_windows, self.Zorig.shape[0], self.max_kine), self.no_spike_value, dtype=np.float32)
        # Loop down each neuron, muscle, assign data to main arrays
        if self.use_ISI:
            for i in range(len(self.Xtimes)):
                column_inds = monotonic_repeats_to_ranges(window_inds_x[i])
                firstval = self.Xtimes[i][0] - self.window_times[window_inds_x[i]][0]
                Xmain[window_inds_x[i],i,column_inds] = np.insert(np.diff(self.Xtimes[i] - self.window_times[window_inds_x[i]]), 0, firstval)
                first = column_inds == 0
                Xmain[window_inds_x[i][first], i, column_inds[first]] = self.Xtimes[i][first] - self.window_times[window_inds_x[i][first]]
            for i in range(len(self.Ytimes)):
                column_inds = monotonic_repeats_to_ranges(window_inds_y[i])
                firstval = self.Ytimes[i][0] - self.window_times[window_inds_y[i]][0]
                Ymain[window_inds_y[i],i,column_inds] = np.insert(np.diff(self.Ytimes[i] - self.window_times[window_inds_y[i]]), 0, firstval)
                first = column_inds == 0
                Ymain[window_inds_y[i][first], i, column_inds[first]] = self.Ytimes[i][first] - self.window_times[window_inds_y[i][first]]
        else:    
            for i in range(len(self.Xtimes)):
                column_inds = monotonic_repeats_to_ranges(window_inds_x[i])
                Xmain[window_inds_x[i],i,column_inds] = self.Xtimes[i] - self.window_times[window_inds_x[i]]
            for i in range(len(self.Ytimes)):
                column_inds = monotonic_repeats_to_ranges(window_inds_y[i])
                Ymain[window_inds_y[i],i,column_inds] = self.Ytimes[i] - self.window_times[window_inds_y[i]]
        # Assign kinematics data
        # Have to interpolate if time offset applied
        mask = self.kine_valid_window[window_inds_z]
        column_inds = monotonic_repeats_to_ranges(window_inds_z)[mask]
        if time_offset > 0:
            for i in range(self.Zorig.shape[0]):
                Zmain[window_inds_z[mask],i,column_inds] = np.interp(self.Ztimes[mask] + time_offset, self.Ztimes[mask], self.Zorig[i,mask])
        else:
            Zmain[window_inds_z[mask],:,column_inds] = np.transpose(self.Zorig[:,mask])
        # Trim windows that aren't valid (too short, in between bouts, etc)
        # Note that there can be windows that are valid for neurons & muscles but not kinematics!
        Xmain = Xmain[self.valid_window,:,:]
        Ymain = Ymain[self.valid_window,:,:]
        Zmain = Zmain[self.valid_window,:,:]
        self.window_times = self.window_times[self.valid_window]
        self.kine_valid_window = self.kine_valid_window[self.valid_window]
        self.n_windows = len(self.window_times)
        # If only keeping windows with kinematics trim further
        if self.only_valid_kinematics:
            Xmain = Xmain[self.kine_valid_window,:,:]
            Ymain = Ymain[self.kine_valid_window,:,:]
            Zmain = Zmain[self.kine_valid_window,:,:]
            self.window_times = self.window_times[self.kine_valid_window]
            self.kine_valid_window = self.kine_valid_window[self.kine_valid_window]
            self.n_windows = len(self.window_times)
        # Trim even further if using kinematics to only get flapping periods
        if self.only_flapping:
            maxvals = Zmain[:,[0],:].max(axis=2)
            minvals = Zmain[:,[0],:].min(axis=2)
            valid = (maxvals - minvals).flatten() > 0.4
            Xmain = Xmain[valid,:,:]
            Ymain = Ymain[valid,:,:]
            Zmain = Zmain[valid,:,:]
            self.window_times = self.window_times[valid]
            self.n_windows = len(self.window_times)
        # Convert to tensor, move to device. Make copies that noise will be applied on
        self.Xmain = torch.tensor(Xmain, device=device)
        self.Ymain = torch.tensor(Ymain, device=device)
        self.Zmain = torch.tensor(Zmain, device=device)
        self.X = self.Xmain.detach().clone()
        self.Y = self.Ymain.detach().clone()
        self.Z = self.Zmain.detach().clone()
        # Pre-compute mask of where actual spikes are
        self.spike_mask_x = torch.nonzero(self.Xmain != self.no_spike_value, as_tuple=True)
        self.spike_mask_y = torch.nonzero(self.Ymain != self.no_spike_value, as_tuple=True)
        # Pre-allocate noise tensor, number of spikes to avoid repeated allocations/operations when applying noise
        self.noise_buffer_x = torch.empty(len(self.spike_mask_x[0]), device=device, dtype=self.X.dtype)
        self.noise_buffer_y = torch.empty(len(self.spike_mask_y[0]), device=device, dtype=self.Y.dtype)
    
    def read_data(self, base_name, sample_rate=30000, neuron_label_filter=None):
        """
        Processes spike data from 3 .npz files into neural (X) and muscle (Y) activity tensors.
        """
        # Construct file paths
        data_file = f"{base_name}_data.npz"
        labels_file = f"{base_name}_labels.npz"
        bouts_file = f"{base_name}_bouts.npz"
        kine_file = f"{base_name}_kinematics.npz"
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
        
        # Load bout start and end indices, rescale to seconds. (-1 is b/c coming from 1-index Julia)
        bouts = np.load(bouts_file)
        starts, ends = bouts['starts'], bouts['ends']
        self.bout_starts = (starts - 1) / sample_rate
        self.bout_ends = (ends - 1) / sample_rate

        # Load kinematics data
        if not os.path.exists(kine_file):
            raise FileNotFoundError(f"This moth has no kinematics data found")
        kine_data = np.load(kine_file)
        kine_names = list(kine_data.keys())
        angle_names = [side + angle for side, angle in product(['L', 'R'], ['phi', 'theta', 'alpha'])]
        point_names = set(kine_names).difference(set(angle_names + ['index']))
        # Allocate matrix to hold everything, fill main angles first
        self.Ztimes = kine_data['index'] / sample_rate
        self.Zorig = np.zeros((len(kine_names)-1, self.Ztimes.shape[0]), dtype=np.float32)
        for i,key in enumerate(angle_names):
            self.Zorig[i,:] = kine_data[key]
        for i,key in enumerate(point_names):
            self.Zorig[i+6,:] = kine_data[key]
        self.kine_names = angle_names + list(point_names)
        
        # Separate neurons and muscles, applying filtering if needed
        self.neuron_labels, self.muscle_labels = [], []
        for unit in units:
            if unit.isnumeric():  # Numeric labels are neurons
                if labels:
                    label = labels.get(unit, None)
                    if neuron_label_filter is None or label == neuron_label_filter:
                        self.neuron_labels.append(unit)
                else:
                    self.neuron_labels.append(unit)  # No filtering if no labels file
            else:  # Alphabetic labels are muscles
                self.muscle_labels.append(unit)
        # Make list of neuron (X) and muscle (Y) spike times
        self.Xtimes, self.Ytimes = [], []
        for unit in self.neuron_labels:
            self.Xtimes.append(np.array(data[unit] / sample_rate))
        for unit in self.muscle_labels:
            self.Ytimes.append(np.array(data[unit] / sample_rate))
        # Close all files
        data.close()
        bouts.close()
        labels_data.close()
        kine_data.close()


def create_data_split(dataset, batch_size, train_fraction=0.95, eval_fraction=None, subset_indices=None):
    """
    Creates train loader and test/eval data views
    Args:
        dataset (FullAndBatchedDataset): The dataset containing all data
        train_fraction (float): Fraction of batches to use for training
    Returns:
        tuple: (train_loader, test_indices, eval_indices)
    """
    num_windows = dataset.n_windows if subset_indices is None else len(subset_indices)
    # Generate train/test splits
    train_size = int(train_fraction * num_windows)
    # Create train/test/eval indices, separate eval set with different random indices
    if subset_indices is None:
        traintest_indices = torch.randperm(dataset.n_windows)
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
