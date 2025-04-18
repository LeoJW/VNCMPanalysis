#!/usr/bin/env python3
import os
from pathlib import Path

# Set the parent directory (modify this if needed)
parent_dir = Path('E:/VNCMP')  # Current directory by default

# Loop through all subdirectories of PARENT_DIR
for date_dir in parent_dir.iterdir():
    if not date_dir.is_dir():
        continue
    # Loop through all Camera directories inside the date directory
    for camera_dir in date_dir.glob('Camera*'):
        if camera_dir.is_dir():
            # Rename each .mp4 file
            for mp4_file in camera_dir.glob('*.mp4'):
                new_name = f"{mp4_file.stem}_{date_dir.name}_{camera_dir.name}{mp4_file.suffix}"
                if (camera_dir / new_name).exists() or (mp4_file.stem.count('_') > 1):
                    continue
                mp4_file.rename(camera_dir / new_name)