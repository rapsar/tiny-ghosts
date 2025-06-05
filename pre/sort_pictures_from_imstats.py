#!/usr/bin/env python3

"""
sort_pictures_from_imstats.py

Usage (in VS Code as separate cells):
1. Adjust `npz_path` to point to your image_stats.npz, then run Cell 1 to visualize the histogram.
2. Tweak the threshold values in Cell 2, set `dest_folder`, and run Cell 2 to sort images accordingly.

Loads precomputed image statistics from a NumPy .npz archive (produced by calculate_image_stats.py),
displays a 2D histogram of mean red brightness vs. (std_red/mean_red) on a log scale,
and then assigns each image to one of four categories (days, dusk, dark, null) based on
user‐defined thresholds. Finally, it copies or symlinks each image into
dest_folder/<category>/<date_folder>/ for downstream processing.

"""

# %% Load image_stats.npz and plot histogram

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# path .npz file
npz_path = "/path/to/image_stats.npz"

# load
data = np.load(npz_path, allow_pickle=True)

avg_red = data['avg_red']
std_red = data['std_red']

# std over mean
ratio = np.zeros_like(std_red)
nonzero = avg_red > 0
ratio[nonzero] = std_red[nonzero] / avg_red[nonzero]

# bins
bins_x = np.arange(0, 32, 0.1)
bins_y = np.arange(0, 0.5, 0.005)

plt.figure(figsize=(8, 6))
h = plt.hist2d(
    avg_red,
    ratio,
    bins=[bins_x, bins_y],
    cmap="turbo",
    norm=LogNorm(vmin=1)
)
plt.colorbar(h[3], label="count")
plt.xlabel("mean picture brightness")
plt.ylabel("std / mean")
plt.title("histogram of image statistics")
plt.tight_layout()
plt.show()


# %% Categorize and move files
# use histogram above to adjust thresholds

# thresholds
min_max_brightness = 50
max_avg_brightness = 8
max_s_m_threshold  = 0.25

import os
import numpy as np
import shutil

# destination folder
dest_folder = "/path/to/destinationFolder"

filenames    = data['filenames']
folders      = data['folders']
is_grayscale = data['is_grayscale']
avg_red      = data['avg_red']
std_red      = data['std_red']
max_red      = data['max_red']

# compute boolean masks
null_TF = max_red < min_max_brightness              # pictures with no bright pixel (> min_max) so no flash
days_TF = (~null_TF) & (~is_grayscale)              # pictures in color, indicating high brightness
dark_TF = (                                         # pictures almost completely dark (low std/mean)
    (avg_red < max_avg_brightness)
    & (ratio < max_s_m_threshold)
    & (~null_TF)
    & (~days_TF)
)
dusk_TF = (~null_TF) & (~days_TF) & (~dark_TF)      # dark but not too dark pictures

categories = {
    "days": np.where(days_TF)[0],
    "dusk": np.where(dusk_TF)[0],
    "dark": np.where(dark_TF)[0],
    "null": np.where(null_TF)[0],
}

# Create destination categories
for cat in categories:
    os.makedirs(os.path.join(dest_folder, cat), exist_ok=True)

# Copy or symlink files based on category
for cat, indices in categories.items():
    for idx in indices:
        src_folder = folders[idx]
        fname      = filenames[idx]
        date_folder = os.path.basename(src_folder.rstrip(os.sep))
        dest_subfolder = os.path.join(dest_folder, cat, date_folder)
        os.makedirs(dest_subfolder, exist_ok=True)

        src_path = os.path.join(src_folder, fname)
        dest_path = os.path.join(dest_subfolder, fname)

        try:
            # If source is symlink, replicate; otherwise copy
            if os.path.islink(src_path):
                link_target = os.readlink(src_path)
                os.symlink(link_target, dest_path)
            else:
                shutil.copy2(src_path, dest_path)
        except Exception as e:
            print(f"[Warning] Failed to copy {src_path} → {dest_path}: {e}")

# Write thresholds.txt
thresholds_txt = os.path.join(dest_folder, "thresholds.txt")
with open(thresholds_txt, "w") as f:
    f.write(f"min_max_brightness = {min_max_brightness}\n")
    f.write(f"max_avg_brightness = {max_avg_brightness}\n")
    f.write(f"max_s/m_threshold = {max_s_m_threshold}\n")

print("Done moving/copying files into categories.")
# %%
