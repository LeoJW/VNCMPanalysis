import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, io
import os
import shutil
from open_ephys.analysis import Session
from functions import *


motor_program_dir = os.path.normpath('/Users/leo/Desktop/ResearchPhD/VNCMP/localdata/motor_program')

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
    # Read open-ephys data
    # Assume only one record node, take first
    rootsession = Session(found_directory)
    session = rootsession.recordnodes[0]
    # Loop over recordings (actually called experiments in open-ephys directory tree)
    trial_counter = 1
    for i in range(len(session.recordings)):
        # Check if there's a NI-DAQ continuous channel
        source_names = [s.metadata['source_node_name'] for s in session.recordings[i].continuous]
        if 'NI-DAQmx' not in source_names:
            print('Cannot find NI-DAQmx continuous source for ' + start_dir)
            continue
        ind = source_names.index('NI-DAQmx')
        fs = session.recordings[i].continuous[ind].metadata['sample_rate']
        # Create filters
        threshold_hpf = signal.butter(3, [threshold_hp_cutoff], btype="highpass", fs=fs, output='sos')
        general_bpf = signal.butter(3, general_bp_cutoff, btype='bandpass', fs=fs, output='sos')
        # Determine how many chunks needed
        exp_len_samples = session.recordings[i].continuous[ind].samples.shape[0]
        nchunks = np.ceil(exp_len_samples / chunk_size).astype(int)
        # Load in chunks, for each detect activity as passing threshold based on noise floor
        has_activity = np.zeros(nchunks, dtype='bool')
        start_times = np.zeros(nchunks)
        print('Scanning chunks for activity...')
        for chunk in range(nchunks-1):
            # Load data of chunk
            data = session.recordings[i].continuous[ind].get_samples(
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
            start_times[chunk] =  session.recordings[i].continuous[ind].timestamps[chunk_size * (chunk-1)]
            # If this chunk had activity, save to file as a "trial"
            if not chunk_active:
                continue
            data = np.hstack((np.reshape(np.arange(0, data.shape[0]) / fs, (-1,1)), data))
            channel_names = np.array(
                    ['time', 'RAX', 'RBA', 'RSA', 'RDVM', 'LAX', 'LBA', 'LSA', 'LDVM', 'RDLM', 'LDLM'], 
                    dtype=object)
            mdict = {'channelNames' : channel_names, 'data' : data}
            file_name = 'EMG_experiment' + str(session.recordings[i].experiment_index) + '_' + str(trial_counter).zfill(3) + '.mat'
            io.savemat(os.path.join(found_directory, sortfiles_dir, file_name), mdict)
            print(str(trial_counter).zfill(3))
            trial_counter += 1

filename = '/Users/leo/Desktop/ResearchPhD/VNCMP/localdata/motor_program/2023-05-24_13-42-33'
rootsession = Session(filename)
session = rootsession.recordnodes[0]
exp_len_samples = session.recordings[0].continuous[0].samples.shape[0]
data = session.recordings[0].continuous[0].get_samples(
                start_sample_index=0, 
                end_sample_index=exp_len_samples,
                selected_channels=np.array([3]))
plt.plot(data)
plt.show()
# plt.figure()
# for j in range(data.shape[1]):
#     plt.plot(data[:,j] / np.ptp(data[:,j]) + j)
# plt.show()


# threshold_hpf = signal.butter(6, [2000], btype="highpass", fs=fs, output='sos')
# # # w, h = signal.sosfreqz(threshold_hpf, worN=10000)
# # # db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
# # # plt.plot(w * fs / (2 * np.pi), db)
# # filtsos = signal.butter(6, [10], btype="highpass", fs=fs, output='sos')
# x = signal.sosfilt(threshold_hpf, data[:,0])
# fft = np.fft.rfft(x)
# psd = (np.abs(fft) ** 2) #/ len(data)
# freqs = np.fft.fftfreq(len(x), 1 / fs)
# idx = np.argsort(freqs)
# plt.plot(freqs[idx], psd[idx])
# # f, Pxx_den = signal.welch(data[:,0], fs, nperseg=1024)
# # plt.semilogy(f, Pxx_den)
# # plt.show()