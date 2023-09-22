import os

main_path = '/Volumes/Mount Evans/VNCMP/'#'05-19-2023/FASTEC-IL4-251_000000/img0000'

dates = [
    '05-19-2023',
    '05-20-2023',
    '05-22-2023',
    '05-24-2023',
    '05-24-2023-01',
    '05-25-2023',
    '05-25-2023-01',
    '05-25-2023-02',
    '05-26-2023',
    '05-26-2023-01',
    '05-27-2023',
    '05-27-2023-01'
]

for d in dates:
    main_folder = os.path.join(main_path, d)
    camera_folders = [f for f in os.listdir(main_folder) if ("charuco" not in f) and ("plumb" not in f)]
    for c in camera_folders:
        camera_folder = os.path.join(main_folder, c)
        img_dirs = os.listdir(camera_folder)
        img_dirs = [f for f in img_dirs if os.path.isdir(os.path.join(camera_folder,f))]
        for i in img_dirs:
            dir = os.path.join(camera_folder, i) + '/'
            os.chdir(dir)
            print(dir)
            image_files = sorted(os.listdir(dir))
            image_files = [f for f in image_files if '._' not in f]
            first_frame = str(int(image_files[0][0:-4]))
            if os.path.isfile('output.mp4'):
                command = "mv output.mp4 ../" + i + ".mp4"
                print(command)
                os.system(command)
            elif not os.path.isfile(os.path.join(camera_folder, i + ".mp4")):
                command = "ffmpeg -y -framerate 24 -start_number " + first_frame + " -i %7d.jpg -c:v libx264 -crf 30 output.mp4"
                print(command)
                os.system(command)