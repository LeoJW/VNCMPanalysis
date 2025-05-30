"""
Version for pre-2025 motor program data. 
Use split_motor_program_fullOpenEphys.py for all recordings taken using intan headstage for EMG
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from scipy import signal, io
from open_ephys.analysis import Session
from functions import *

motor_program_dir = os.path.normpath('/Users/leo/Desktop/ResearchPhD/VNCMP/localdata/pre_2025/motor_program')
# motor_program_dir = os.path.normpath('/Volumes/PikesPeak/Open Ephys/good_data')

# Controls
overwrite_previous = True

# Settings
chunk_size = 750000 # samples

threshold_times_pkpk = 1
threshold_hp_cutoff = 2000 # hz
general_bp_cutoff = [10, 1000] # hz

channel_inds = np.hstack((np.arange(8), [16,17]))

moth_files = [f for f in os.listdir(motor_program_dir) if not f.startswith('.')]

for start_dir in moth_files:
    print('--------')
    print(start_dir)
    # Find level of nesting that Record Node files start at
    found_directory = None
    for root, dirs, files in os.walk(os.path.join(motor_program_dir, start_dir)):
        if any('Record Node' in f for f in dirs):
            found_directory = root
            break
    if found_directory is None:
        print('No open-ephys directory with Record Nodes found in ' + start_dir)
        continue
    
    # Set up output folder, skip this moth if we don't want to overwrite
    sortfiles_dir = os.path.join(found_directory, start_dir + '_spikesort')
    if os.path.isdir(sortfiles_dir) and not overwrite_previous:
        continue
    elif not os.path.isdir(sortfiles_dir):
        os.mkdir(sortfiles_dir)
    # If the folder exists and we want overwrite, delete all files 
    else:
        for filename in os.listdir(sortfiles_dir):
            file_path = os.path.join(sortfiles_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    trial_start_times_file = open(os.path.join(sortfiles_dir, 'trial_start_times.txt'), 'w')
    trial_start_times_file.write('file name, start sample index, start time (s)\n')
    experiment_start_times_file = open(os.path.join(sortfiles_dir, 'experiment_start_times.txt'), 'w')
    experiment_start_times_file.write('experiment, time (milliseconds since midnight Jan 1st 1970 UTC):\n')
    # Read open-ephys data
    # Assume only one record node, take first
    rootsession = Session(found_directory)
    session = rootsession.recordnodes[0]
    # Loop over recordings (actually called experiments in open-ephys directory tree)
    trial_counter = 1
    for exp in session.recordings:
        print('Experiment ' + str(exp.experiment_index))
        # Check if there's a NI-DAQ continuous channel
        source_names = [s.metadata['source_node_name'] for s in exp.continuous]
        if 'NI-DAQmx' not in source_names:
            print('Cannot find NI-DAQmx continuous source for ' + start_dir)
            continue
        ind = source_names.index('NI-DAQmx')
        fs = exp.continuous[ind].metadata['sample_rate']
        # Create filters
        threshold_hpf = signal.butter(3, [threshold_hp_cutoff], btype="highpass", fs=fs, output='sos')
        general_bpf = signal.butter(3, general_bp_cutoff, btype='bandpass', fs=fs, output='sos')
        # Determine how many chunks needed
        exp_len_samples = exp.continuous[ind].samples.shape[0]
        nchunks = np.ceil(exp_len_samples / chunk_size).astype(int)
        # Load in chunks, for each detect activity as passing threshold based on noise floor
        for chunk in range(nchunks-1):
            # Load data of chunk
            data = exp.continuous[ind].get_samples(
                start_sample_index=chunk_size * chunk, 
                end_sample_index=chunk_size * (chunk + 1),
                selected_channels=channel_inds)
            # In first chunk, set threshold amplitude based on average peak-peak amplitude of all muscles above 2000Hz
            # Decent metric as almost all of the EMG signal is below 2000
            if chunk == 0:
                pkpk = np.zeros(10)
                for channel in range(10):
                    pkpk[channel] = np.ptp(signal.sosfilt(threshold_hpf, data[:,channel]))
                threshold = threshold_times_pkpk * np.mean(pkpk)
            chunk_active = np.any(signal.sosfilt(general_bpf, data, axis=0) > threshold)
            # If this chunk had activity, save to file as a "trial"
            if not chunk_active:
                continue
            data = np.hstack((np.reshape(np.arange(0, data.shape[0]) / fs, (-1,1)), data))
            channel_names = np.array(
                    ['time', 'RAX', 'RBA', 'RSA', 'RDVM', 'LAX', 'LBA', 'LSA', 'LDVM', 'RDLM', 'LDLM'], 
                    dtype=object)
            # NOTE: I suspect LBA and LAX are switched here, based on end data. Must have been switched on rig itself
            mdict = {'channelNames' : channel_names, 'data' : data}
            file_name = 'EMG_experiment' + str(exp.experiment_index) + '_' + str(trial_counter).zfill(3) + '.mat'
            io.savemat(os.path.join(found_directory, sortfiles_dir, file_name), mdict)
            # Write to metadata file of trial start times
            start_index = chunk_size * chunk
            start_time = exp.continuous[ind].timestamps[start_index]
            trial_start_times_file.write(file_name + f', {start_index}, {start_time}' + '\n')
            print(str(trial_counter).zfill(3))
            trial_counter += 1
        # Write to metadata file of experiment start times
        with open(os.path.join(exp.directory, 'sync_messages.txt'), 'r') as f:
            timestamp = f.readline().strip('\n').split(': ')[1]
        experiment_start_times_file.write('experiment' + str(exp.experiment_index) + ', ' + timestamp + '\n')
    trial_start_times_file.close()
    experiment_start_times_file.close()