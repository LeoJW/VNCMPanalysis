import os
import subprocess
import shutil
from pathlib import Path

"""Compress a video file using ffmpeg and save to the output location"""
def compress_video(input_file, output_file):
    # ffmpeg command for compression - adjust parameters as needed for your quality/size requirements
    # This example uses a lower bitrate and crf value for compression
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'h264_nvenc',
        '-preset', 'p7',  # Encoding speed/compression tradeoff (set to slowest acceptable)
        '-qp', '21',  # Compression quality (23 is default, higher means more compression)
        '-y',  # Overwrite output files
        output_file
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Compressed: {input_file} -> {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error compressing {input_file}: {e}")
        return False

"""Process all mp4 files in the folder structure"""
def process_folder(root_folder):
    # Convert to absolute path
    root_folder = os.path.abspath(root_folder)
    # Walk through directory structure
    for moth_dir in os.listdir(root_folder):
        # Filter for mp4 files
        files = os.listdir(os.path.join(root_folder, moth_dir))
        mp4_files = [f for f in files if f.lower().endswith('.mp4')]
        if mp4_files:
            # Create compressed subfolder
            compressed_dir = os.path.join(root_folder, moth_dir, "compressed")
            os.makedirs(compressed_dir, exist_ok=True)
            # Process each mp4 file
            for mp4_file in mp4_files:
                input_path = os.path.join(root_folder, moth_dir, mp4_file)
                output_path = os.path.join(compressed_dir, mp4_file)
                # Skip if the file already exists in compressed folder
                if os.path.exists(output_path):
                    print(f"Skipping (already exists): {output_path}")
                    continue
                compress_video(input_path, output_path)

if __name__ == "__main__":
    # Input the root folder path here
    # root_folder = input("Enter the path to your root folder: ")
    root_folder = os.path.abspath('E:/VNCMP/')
    
    # Check if folder exists
    if not os.path.isdir(root_folder):
        print(f"Error: {root_folder} is not a valid directory")
        exit(1)
    # Verify ffmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not in PATH")
        print("Please install ffmpeg before running this script")
        exit(1)
    
    # Process the folder structure
    process_folder(root_folder)
    print("Compression complete!")