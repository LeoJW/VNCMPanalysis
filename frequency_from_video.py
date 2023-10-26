import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal

main_path = '/Volumes/BlueSky/VNCMP/'#'05-19-2023/FASTEC-IL4-251_000000/img0000'

load_from_images = False

# dates = [
#     '05-19-2023',
#     '05-20-2023',
#     '05-22-2023',
#     '05-24-2023',
#     '05-24-2023-01',
#     '05-25-2023',
#     '05-25-2023-01',
#     '05-25-2023-02',
#     '05-26-2023',
#     '05-26-2023-01',
#     '05-27-2023',
#     '05-27-2023-01'
# ]
dates = [
    '05-20-2023'
]

fs = 350
hpf = signal.butter(3, 3, btype="highpass", fs=fs, output='sos')

for d in dates:
    main_folder = os.path.join(main_path, d)
    camera_folders = [f for f in os.listdir(main_folder) if ("charuco" not in f) and ("plumb" not in f)]
    # Just take fastec-3
    cam = [s for s in camera_folders if '-3' in s][0]
    camera_folder = os.path.join(main_folder, cam)
    img_dirs = [f for f in os.listdir(camera_folder) if os.path.isdir(os.path.join(camera_folder,f))]
    for chunk in img_dirs:
        if load_from_images:
            dir = os.path.join(camera_folder, chunk) + '/'
            print(dir)
            image_files = [f for f in sorted(os.listdir(dir)) if '._' not in f]
            trace = np.zeros(len(image_files))
            print(chunk)
            for i, image in enumerate(image_files):
                im = Image.open(os.path.join(dir, image))
                pix = im.load()
                trace[i] = np.max(im.crop((349, 0, 350, 820)))
            np.save(os.path.join(camera_folder, chunk), trace)
        else:
            trace = np.load(os.path.join(camera_folder, chunk) + '.npy')
        
        # plt.plot(trace)
        # plt.show()
        tracefilt = signal.sosfilt(hpf, trace)
        plt.plot(np.arange(0,len(tracefilt))/30000, tracefilt)
        # f, Pxx_spec = signal.welch(trace.flatten(), fs, 
        #                    nperseg=4096*10, 
        #                    scaling='density',
        #                    detrend=False)
        # plt.loglog(f, np.sqrt(Pxx_spec))

        # f, t, Sxx = signal.spectrogram(trace, fs)
        # plt.pcolormesh(t, f, Sxx, shading='gouraud')
        # plt.ylabel('Frequency [Hz]')
        plt.ylabel('Trace')
        plt.xlabel('Time [sec]')
        # plt.savefig(os.path.join(camera_folder, chunk) + 'mid_spectrogram.png')
        plt.gcf().savefig(os.path.join(camera_folder, chunk) + 'rawtrace.png')