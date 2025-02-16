import os
import numpy as np
from PIL import Image

def preprocess_image(img):
    """
    Preprocess the image by removing the bottom 140 pixels.

    Args:
        img (PIL.Image.Image): The image to preprocess.

    Returns:
        PIL.Image.Image: The cropped image.
    """
    # Crop the image to remove the bottom 140 pixels
    return img.crop((0, 0, img.width, img.height - 140))

def calculate_brightness(image_path):
    """
    Calculate the average brightness and standard deviation of the green channel of the image after preprocessing.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: The average brightness and standard deviation of the green channel.
    """
    with Image.open(image_path) as img:
        # Preprocess the image (crop it)
        img = preprocess_image(img)
        # Convert the image to a numpy array and extract the green channel
        green_channel = np.array(img)[:, :, 1]
        # Compute and return the average brightness and standard deviation of the green channel
        return np.mean(green_channel), np.std(green_channel)

def create_output_folders(output_folder):
    """
    Create the required output folders for sorting images.

    Args:
        output_folder (str): Path to the main output folder.

    Returns:
        tuple: Paths to the 'dark' and 'dusk' folders.
    """
    # Ensure the root output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define paths for dark and dusk folders, including their '_flash' subfolders
    dark_folder = os.path.join(output_folder, "dark")
    dark_flash_folder = os.path.join(dark_folder, "_flash")
    dusk_folder = os.path.join(output_folder, "dusk")
    dusk_flash_folder = os.path.join(dusk_folder, "_flash")

    # Create the folders if they do not exist
    os.makedirs(dark_flash_folder, exist_ok=True)
    os.makedirs(dusk_flash_folder, exist_ok=True)

    return dark_folder, dusk_folder

def calculate_thresholds(brightness_values):
    """
    Calculate the brightness and ratio thresholds for sorting images.

    Args:
        brightness_values (list): List of brightness values for all images.

    Returns:
        tuple: The brightness threshold and ratio threshold.
    """
    brightness_threshold = 26
    ratio_threshold = 0.6
    return brightness_threshold, ratio_threshold

def sort_images(input_folder, output_folder):
    """
    Sort images into 'dark' and 'dusk' categories based on brightness.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where sorted images will be stored.
    """
    # Create the output folders
    dark_folder, dusk_folder = create_output_folders(output_folder)

    mean_threshold, ratio_threshold = calculate_thresholds([])

    # Iterate through files in the input folder
    for filename in os.listdir(input_folder):
        # Process only files that match the naming pattern
        if filename.startswith("DSCF") and filename.endswith(".JPG"):
            image_path = os.path.join(input_folder, filename)
            # Calculate brightness for the current image
            brightness, std_dev = calculate_brightness(image_path)
            
            # Determine target folder based on brightness and std/mean ratio
            if brightness < mean_threshold and (std_dev / brightness) < ratio_threshold:
                symlink_path = os.path.join(dark_folder, filename)
            else:
                symlink_path = os.path.join(dusk_folder, filename)

            # Create a symlink to the image in the appropriate folder
            if not os.path.exists(symlink_path):
                os.symlink(image_path, symlink_path)

def main():
    """
    Main function to parse arguments and initiate the sorting process.
    """
    import argparse

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Sort images into dark and dusk categories.")
    parser.add_argument("-i", "--input", required=True, help="Input folder containing images.")
    parser.add_argument("-o", "--output", required=True, help="Output folder for sorted images.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Retrieve input and output folder paths from arguments
    input_folder = args.input
    output_folder = args.output

    # Perform the image sorting process
    sort_images(input_folder, output_folder)

if __name__ == "__main__":
    main()
