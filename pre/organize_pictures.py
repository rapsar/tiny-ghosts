#!/usr/bin/env python3
"""
organize_pictures.py

See __main__ at the bottom to enter input and output folders.

Organizes a DCIM directory (with nested subfolders) into a clean folder of date-based subfolders (YYYYMMDD),
using symlinks or copies as specified.
Takes as input a DCIM folder containing subfolders (e.g., 100MEDIA) and outputs a destination folder
where each subfolder is named by the photo date (YYYYMMDD).
Photos are prefixed with corresponding YYYYMMDDThhmmss_

Initial folder structure:
    source_base/
    ├── 100MEDIA/
    │   ├── DSCF0001.JPG
    │   ├── DSCF0002.JPG
    │   └── ...
    ├── 101MEDIA/
    │   ├── photoA.jpg
    │   └── ...
    └── ...

Output folder structure:
    destination_folder/
    ├── YYYYMMDD/
    │   ├── YYYYMMDDThhmmss_DSCF0001.jpg
    │   ├── YYYYMMDDThhmmss_DSCF0002.jpg
    │   └── ...
    ├── YYYYMMDD/
    │   ├── YYYYMMDDThhmmss_photoA.jpg
    │   └── ...
    └── ...

RS, 05/2025
"""
import os
import shutil
import datetime
import subprocess
from PIL import Image, ExifTags

def get_datetime_from_exif(image_path):
    """
    Extracts DateTimeOriginal from the image's EXIF data and returns it
    formatted as 'yyyymmddTHHMMSS' (ISO 8601). If the tag is missing or an
    error occurs, returns None.
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data:
                # Loop through EXIF tags looking for DateTimeOriginal
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == "DateTimeOriginal":
                        # EXIF format: "YYYY:MM:DD HH:MM:SS"
                        dt = datetime.datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                        return dt.strftime("%Y%m%dT%H%M%S"), dt.strftime("%Y%m%d")
    except Exception as e:
        print(f"Error reading EXIF from {image_path}: {e}")
    return None, None

def list_directory(path):
    """
    Uses subprocess to list a directory's contents. This works around issues
    with certain external drives where os.listdir() fails.
    """
    try:
        result = subprocess.run(['ls', path], capture_output=True, text=True, check=True)
        # Split the output by lines and filter out empty lines
        return [item for item in result.stdout.split('\n') if item]
    except subprocess.CalledProcessError as e:
        print(f"Error listing directory using subprocess: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return []

def process_photos(source_base, destination_folder, use_symlinks=False, use_subfolders=True):
    """
    Walks through subdirectories of source_base (looking for directories ending with 'MEDIA'),
    processes each .JPG file to extract EXIF date/time, and either copies the file or creates a symlink
    in destination_folder with the date/time prepended to the original filename.
    Organizes images into subfolders based on their date ('YYYYMMDD')
    """
    # Ensure destination exists
    os.makedirs(destination_folder, exist_ok=True)
    
    # List the entries in the source_base directory using subprocess
    base_entries = list_directory(source_base)
    if not base_entries:
        print(f"No entries found in base directory '{source_base}' or error occurred")
        return
    
    for folder in base_entries:
        folder_path = os.path.join(source_base, folder)
        if os.path.isdir(folder_path) and folder.endswith("MEDIA"):
            print(f"Processing folder: {folder_path}")
            
            # List files in the MEDIA folder using subprocess
            media_entries = list_directory(folder_path)
            if not media_entries:
                print(f"No entries found in media directory '{folder_path}' or error occurred")
                continue
                
            for filename in media_entries:
                if filename.lower().endswith(".jpg"):
                    file_path = os.path.join(folder_path, filename)
                    dt_str, date_folder = get_datetime_from_exif(file_path)
                    
                    if use_subfolders and date_folder:
                        subfolder_path = os.path.join(destination_folder, date_folder)
                        os.makedirs(subfolder_path, exist_ok=True)  # Ensure the date subfolder exists
                        #new_filename = f"{dt_str}_{filename}" if dt_str else filename
                    else:
                        print(f"EXIF data not found for '{file_path}'. Using original filename and placing in main folder.")
                        subfolder_path = destination_folder
                        #new_filename = filename
                    
                    new_filename = f"{dt_str}_{filename}" if dt_str else filename
                    dest_file = os.path.join(subfolder_path, new_filename)
                    try:
                        if use_symlinks:
                            os.symlink(file_path, dest_file)
                        else:
                            shutil.copy2(file_path, dest_file)
                        # print(f"Processed '{file_path}' to '{dest_file}'")
                    except Exception as e:
                        print(f"Error processing '{file_path}' to '{dest_file}': {e}")

if __name__ == "__main__":
    # Replace with actual path
    # the DCIM/ containing the 100MEDIA subfolders
    source_base = r'/Volumes/My Passport/Bw wickershamorum 2024/Muleshoe/Hot Springs at Stone Cabin/DCIM'
    # the folder to store organized data, typically /data/site_name
    destination_folder = r"/Users/rss367/Desktop/2024bww/Muleshoe/data/HotSpringsStoneCabin"
    use_symlinks = True             # True: copies symlinks instead of actual files. Set to False to copy files instead
    use_subfolders = True           # True: arranges files by subfolders (useful when working with 100k files otherwise explorers lose it); False: puts all files together
    
    process_photos(source_base, destination_folder, use_symlinks, use_subfolders)
