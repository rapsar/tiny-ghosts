#!/usr/bin/env python3
"""
calculate_image_stats.py

Calculate image statistics (avg, std, is_grayscale, max) for all *DSCF*.JPG files 
in a given folder. If no matching files are found in the top-level folder, it looks 
for immediate subfolders and processes any *DSCF*.JPG files inside them.

Usage:
    python calculate_image_stats.py --input /path/to/inputFolder

Outputs a Python dictionary 'imstats' with fields:
    - input_folder: str
    - files: list of dicts, each with 'folder' and 'name' keys
    - is_grayscale: NumPy array of bools
    - avg_red:      NumPy array of floats
    - std_red:      NumPy array of floats
    - max_red:      NumPy array of floats

It also prints progress to the console and timing information.
"""

import os
import argparse
import time
import numpy as np
from PIL import Image
from tqdm import tqdm

def find_dscf_files(input_folder):
    """
    Returns a list of dicts with keys 'folder' and 'name' for all files matching 
    '*DSCF*.JPG' in input_folder. If none are found, checks immediate subfolders.
    """
    matches = []

    # First, look for *DSCF*.JPG in top-level input_folder
    for fname in os.listdir(input_folder):
        if fname.upper().startswith('DSCF') or 'DSCF' in fname.upper():
            # Ensure it ends in .JPG (case-insensitive)
            if fname.upper().endswith('.JPG'):
                matches.append({'folder': input_folder, 'name': fname})

    if matches:
        return matches

    # If none found, scan immediate subfolders (not recursive beyond one level)
    for entry in os.listdir(input_folder):
        subpath = os.path.join(input_folder, entry)
        if os.path.isdir(subpath) and not entry.startswith('.'):
            for fname in os.listdir(subpath):
                if fname.upper().startswith('DSCF') or 'DSCF' in fname.upper():
                    if fname.upper().endswith('.JPG'):
                        matches.append({'folder': subpath, 'name': fname})

    return matches


def calculate_image_stats(input_folder):
    """
    For each file in the list returned by find_dscf_files(), read the image,
    crop to rows [0:1280), extract the red and green channels, and compute:
      - is_grayscale: True if red == green everywhere
      - mean_red, std_red, max_red (on the red channel, as float)
    Returns a dict 'imstats'.
    """
    imstats = {}
    imstats['input_folder'] = input_folder

    all_files = find_dscf_files(input_folder)
    n_files = len(all_files)

    if n_files == 0:
        print(f"No images found in '{input_folder}' or its immediate subfolders.")
        imstats['files'] = []
        imstats['is_grayscale'] = np.array([], dtype=bool)
        imstats['avg_red']      = np.array([], dtype=float)
        imstats['std_red']      = np.array([], dtype=float)
        imstats['max_red']      = np.array([], dtype=float)
        return imstats

    print(f"Started processing {n_files} files...")
    start_time = time.time()

    # Prepare arrays to hold statistics
    is_gray = np.zeros(n_files, dtype=bool)
    avg_red = np.zeros(n_files, dtype=float)
    std_red = np.zeros(n_files, dtype=float)
    max_red = np.zeros(n_files, dtype=float)

    for i, info in enumerate(tqdm(all_files, desc='Processing images'), start=1):
        full_path = os.path.join(info['folder'], info['name'])
        try:
            # Open image and convert to RGB (in case it's grayscale or CMYK, etc.)
            with Image.open(full_path) as img:
                img = img.convert('RGB')
                arr = np.array(img)

            # Crop to top 1280 rows (if image has at least 1280 rows)
            if arr.shape[0] >= 1280:
                arr = arr[:1280, :, :]
            # else, use the entire image if it's smaller than 1280 rows

            # Extract red and green channels
            red_chan   = arr[:, :, 0]
            green_chan = arr[:, :, 1]

            # Determine if the image is grayscale (red == green everywhere)
            is_gray[i-1] = np.array_equal(red_chan, green_chan)

            # Cast red channel to float64 for statistics
            red_flat = red_chan.astype(np.float64).ravel()

            # Compute statistics on red channel
            avg_red[i-1]  = np.mean(red_flat)
            std_red[i-1]  = np.std(red_flat, ddof=0)  # population std to match MATLAB's std()
            max_red[i-1]  = np.max(red_flat)

        except Exception as e:
            print(f"[Warning] Failed to process '{full_path}': {e}")
            # Leave the default zeros/False in place, but continue

    elapsed = time.time() - start_time
    print(f"Finished processing {n_files} images in {elapsed:.1f} seconds.")

    # Populate return dictionary
    imstats['files']        = all_files
    imstats['is_grayscale'] = is_gray
    imstats['avg_red']      = avg_red
    imstats['std_red']      = std_red
    imstats['max_red']      = max_red

    return imstats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate image statistics for all '*DSCF*.JPG' files in a folder (or its immediate subfolders)."
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help="Path to the input folder containing images"
    )
    parser.add_argument(
        '--output', '-o',
        required=False,
        default=None,
        help="Path to folder where .npz results should be saved (default: same as input folder)"
    )
    args = parser.parse_args()

    target_folder = args.output if args.output else args.input
    os.makedirs(target_folder, exist_ok=True)

    stats = calculate_image_stats(args.input)

    # Extract folders and full paths for each image
    folders = [info['folder'] for info in stats['files']]
    full_paths = [os.path.join(info['folder'], info['name']) for info in stats['files']]
    # Save stats to a NumPy archive for downstream processing
    filenames = [info['name'] for info in stats['files']]
    np.savez(
        os.path.join(target_folder, 'image_stats.npz'),
        filenames=filenames,
        folders=folders,
        full_paths=full_paths,
        is_grayscale=stats['is_grayscale'],
        avg_red=stats['avg_red'],
        std_red=stats['std_red'],
        max_red=stats['max_red']
    )
    print(f"Saved NumPy archive to {os.path.join(target_folder, 'image_stats.npz')}")