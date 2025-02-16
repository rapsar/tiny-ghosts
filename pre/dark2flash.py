import argparse
import os
import cv2
import shutil
import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import cdist
from argparse import Namespace

# Function to preprocess the image
def preprocess_image(image, param):
    # Extract the green channel
    green_channel = image[:, :, 1]

    # Crop the image using the parameters
    cropped_image = green_channel[param.crop_top:-param.crop_bottom, param.crop_left:-param.crop_right]

    # Apply Gaussian blur with the specified radius
    blurred_image = cv2.GaussianBlur(cropped_image, (5, 5), param.gaussian_radius)

    return blurred_image

# Function to process the image
def process_image(image, param):

    # Binarize the image
    _, binary_image = cv2.threshold(image, param.threshold, 255, cv2.THRESH_BINARY)

    # Label connected components in the binary image
    labeled_array, num_features = label(binary_image)

    # Calculate centroids of the labeled regions, ignoring blobs smaller than min_blob_size
    xy_coordinates = []
    for i in range(1, num_features + 1):
        blob_mask = labeled_array == i
        if np.sum(blob_mask) >= param.min_blob_size:  # Blob size filter
            centroid = center_of_mass(blob_mask)
            xy_coordinates.append((int(centroid[1]), int(centroid[0])))  # x, y format

    # If there are more than max_blob blobs, consider the image too noisy
    if len(xy_coordinates) > param.max_blob:
        return []

    return xy_coordinates

# Function to post-process the xy coordinates
def postprocess_coordinates(all_coordinates, param):
    # Flatten all coordinates into a single list with their associated filenames
    flattened_coordinates = []
    for item in all_coordinates:
        filename = item['filename']
        for coord in item['coordinates']:
            flattened_coordinates.append((filename, coord))

    # Calculate distances between all pairs of coordinates
    coordinates_only = np.array([coord[1] for coord in flattened_coordinates])
    distances = cdist(coordinates_only, coordinates_only)

    # Identify and remove coordinates closer than distance_cutoff pixels to another
    to_remove = set()
    num_coords = len(flattened_coordinates)
    for i in range(num_coords):
        for j in range(i + 1, num_coords):
            if distances[i, j] < param.distance_cutoff:
                to_remove.add(i)
                to_remove.add(j)

    # Filter out invalid coordinates
    valid_coordinates = [flattened_coordinates[i] for i in range(num_coords) if i not in to_remove]

    # Group valid coordinates back by filenames
    filenames_with_valid_coords = set(filename for filename, _ in valid_coordinates)

    return list(filenames_with_valid_coords)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process images in a folder of symlinks to detect flashes.")
    parser.add_argument("-i", "--input", required=True, help="Path to the folder containing symlinks to images.")
    parser.add_argument("--threshold", type=int, default=48, help="Threshold value for binarization (default: 48).")
    parser.add_argument("--action", choices=["move", "copy"], default="copy", help="Whether to move or copy images to the '_flash' folder (default: copy).")
    parser.add_argument("--symlinks", action="store_true", help="If set, symlinks will be copied/moved instead of the actual images.")
    args = parser.parse_args()

    input_folder = args.input

    # Define parameters using Namespace
    param = Namespace(
        crop_top=16,
        crop_bottom=144,
        crop_left=16,
        crop_right=16,
        gaussian_radius=2,
        threshold=args.threshold,
        min_blob_size=8,
        max_blob=5,
        distance_cutoff=16
    )

    # Check if the input folder exists
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Create the output folder if it doesn't exist
    flash_folder = os.path.join(input_folder, "_flash")
    os.makedirs(flash_folder, exist_ok=True)

    all_coordinates = []

    # Process each symlink in the input folder
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        # Skip if not a symlink
        if not os.path.islink(filepath):
            continue

        # Read the image
        try:
            real_path = os.readlink(filepath)
            image = cv2.imread(real_path)
            if image is None:
                print(f"Warning: Could not read image '{real_path}'.")
                continue
        except Exception as e:
            print(f"Error reading file '{filepath}': {e}")
            continue

        # Preprocess the image
        preprocessed_image = preprocess_image(image, param)

        # Process the image to extract xy coordinates
        xy_coordinates = process_image(preprocessed_image, param)

        # Append the results to the all_coordinates list
        all_coordinates.append({"filename": filename, "coordinates": xy_coordinates})

    # Post-process the xy coordinates
    valid_images = postprocess_coordinates(all_coordinates, param)

    # Move or copy images or symlinks of valid images to the '_flash' folder
    for filename in valid_images:
        filepath = os.path.join(input_folder, filename)
        real_path = os.readlink(filepath)
        destination_path = os.path.join(flash_folder, filename)

        if args.symlinks:
            if not os.path.exists(destination_path):
                os.symlink(real_path, destination_path)
        else:
            shutil.copy(real_path, destination_path)

        if args.action == "move":
            os.unlink(filepath)

if __name__ == "__main__":
    main()
