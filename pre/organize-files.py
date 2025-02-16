
"""
This script sorts .JPG files by their creation date extracted from metadata.

Usage:
    python sort_files.py --input <input_folder> --output <output_folder> [--symlink]

Arguments:
    --input  : Path to the input folder containing subfolders with .JPG files.
    --output : Path to the destination folder where sorted files will be copied or linked.
    --symlink: Optional flag to create symlinks instead of copying files.

Folder Structure:
    Input folder:
        <input_folder>/
            subfolder1/
                file1.JPG
                file2.JPG
            subfolder2/
                file3.JPG
                ...

    Output folder:
        <output_folder>/
            YYYYMMDD/
                file1.JPG
                file2.JPG
            ...

    - The output folder must already exist.
    - Files are organized into subfolders named by their creation date in YYYYMMDD format.
    - If a file with the same name already exists in the destination folder, it is skipped with a warning.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import argparse
from PIL import Image
from PIL.ExifTags import TAGS

def get_creation_date(file_path):
    try:
        with Image.open(file_path) as img:
            exif_data = img._getexif()
            if exif_data:
                for tag, value in exif_data.items():
                    if TAGS.get(tag) == "DateTimeOriginal":
                        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S").strftime("%Y%m%d")
    except Exception as e:
        print(f"Error reading metadata from {file_path}: {e}")
    return None

def sort_files(input_folder, output_folder, symlink_only=False):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not output_path.exists() or not output_path.is_dir():
        raise ValueError("The output folder must exist and be a directory.")

    for subfolder in input_path.iterdir():
        if subfolder.is_dir():
            for file in subfolder.glob("*.JPG"):
                creation_date = get_creation_date(file)
                if creation_date:
                    destination_folder = output_path / creation_date
                    if not destination_folder.exists():
                        destination_folder.mkdir(exist_ok=True)
                        print(f"Created folder: {destination_folder}")
                    destination_file = destination_folder / file.name
                    if destination_file.exists():
                        print(f"Warning: File {destination_file} already exists. Skipping {file}.")
                        continue
                    if symlink_only:
                        os.symlink(file, destination_file)
                        # print(f"Created symlink for {file} in {destination_folder}")
                    else:
                        shutil.copy(file, destination_file)
                        # print(f"Copied {file} to {destination_folder}")
                else:
                    print(f"Could not determine creation date for {file}")

def main():
    parser = argparse.ArgumentParser(description="Sort .JPG files by creation date.")
    parser.add_argument("--input", required=True, help="Path to the input folder.")
    parser.add_argument("--output", required=True, help="Path to the destination folder.")
    parser.add_argument("--symlink", action="store_true", help="Create symlinks instead of copying files.")

    args = parser.parse_args()
    sort_files(args.input, args.output, symlink_only=args.symlink)

if __name__ == "__main__":
    main()
