# Firefly Flash Detection in Images
# This script processes images to detect firefly flashes using a shell script interface (Llava-cli). 
# Images can be split into patches for analysis, dynamically determined based on the --patch parameter.
# Results are logged, and positive detections are saved to an output folder.

# Usage:
# python script_name.py --input <input_folder> --output <output_folder> --results_file <results_file> --patch <patch_mode>
# Example:
# python script_name.py --input ./images --output ./results --results_file detection_log.txt --patch 4

import os
import subprocess
import shutil
import argparse
import time
from PIL import Image

def run_llava(shell_script_path, image_path):
    # Run the shell script with the image path as an argument
    result = subprocess.run([shell_script_path, image_path], stdout=subprocess.PIPE, text=True)
    
    return result.stdout

def process_output(output):
    # Find the relevant section
    marker = "encode_image_with_clip"
    relevant_output = output[output.find(marker):]
    
    return "yes" in relevant_output.lower()

def extract_relevant_output(output):
    marker = "encode_image_with_clip:"
    try:
        marker_index = output.index(marker) + len(marker)
        return output[marker_index:].strip()  # Strips leading/trailing whitespace
    except ValueError:
        return "Relevant output not found"

def crop_and_split_image(image_path, temp_dir, patch):
    # Open the image
    image = Image.open(image_path)
    cropped_image = image.crop((0, 0, image.width, image.height - 144))

    if patch == 4:
        cropped_image = cropped_image.crop((0, cropped_image.height - 640, cropped_image.width, cropped_image.height))

    base_name = os.path.basename(image_path)
    digits = ''.join(filter(str.isdigit, base_name))[-3:]

    cols = 1 if patch == 1 else (2 if patch == 2 else (3 if patch == 6 else (4 if patch in [4, 8] else 1)))
    rows = 1 if patch in [1, 2, 4] else (2 if patch in [6, 8] else 1)

    patch_width = cropped_image.width // cols
    patch_height = cropped_image.height // rows

    patch_paths = []

    for i in range(cols):
        for j in range(rows):
            left = i * patch_width
            top = j * patch_height
            right = left + patch_width
            bottom = top + patch_height
            patch = cropped_image.crop((left, top, right, bottom))
            patch_path = os.path.join(temp_dir, f"patch_{digits}_{i}_{j}.JPG")
            patch.save(patch_path)
            patch_paths.append(patch_path)

    return patch_paths

def main(image_dir, output_dir, results_file, patch):
    positive_dir = os.path.join(output_dir, f'_positive_llava_patch{patch}')
    os.makedirs(positive_dir, exist_ok=True)

    temp_dir = os.path.join(image_dir, '_temp_patches')
    os.makedirs(temp_dir, exist_ok=True)

    filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg') and f.startswith('DSCF')])

    print(f"Found {len(filenames)} images")

    # Open the results file in append mode
    with open(results_file, 'a') as results_file:
        for filename in filenames:
            image_path = os.path.join(image_dir, filename)
            absolute_image_path = os.path.abspath(image_path)
            print(f"Processing image: {absolute_image_path}")

            patch_paths = crop_and_split_image(image_path, temp_dir, patch)

            for patch_path in patch_paths:
                patch_name = os.path.basename(patch_path)
                print(f"Processing patch: {patch_name}")
                patch_output = run_llava(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../private/run_llava.sh'), patch_path) # replace with your own path and completed script
                
                # Extract relevant output
                relevant_output = extract_relevant_output(patch_output)
                
                print("Output from llava-cli:")
                print(relevant_output)
                
                patch_result = "Positive" if process_output(patch_output) else "Negative"
                results_file.write(f"Image: {absolute_image_path}\nPatch: {patch_name}\nOutput: {relevant_output}\n\n")

                if patch_result == "Positive":
                    shutil.copy(absolute_image_path, positive_dir)
                    break

            for patch_path in patch_paths:
                os.remove(patch_path)
                
            # Add a sleep after processing each image
            time.sleep(1) # who knows if it does anything?

    print("Processing complete.")

if __name__ == "__main__":
    caffeinate_process = subprocess.Popen(["caffeinate", "-ims"])  # Start caffeinate
    try:
        parser = argparse.ArgumentParser(description="Process images for firefly flash detection.")
        parser.add_argument("--input", type=str, required=True, help="Path to the input folder containing images.")
        parser.add_argument("--output", type=str, required=True, help="Path to the output folder for classified images.")
        parser.add_argument("--results_file", type=str, required=True, help="Path to the results file.")
        parser.add_argument("--patch", type=int, default=1, choices=[1, 2, 4, 6, 8], help="Number of patches to split the image into.")
        args = parser.parse_args()

        main(args.input, args.output, args.results_file, args.patch)
    finally:
        caffeinate_process.terminate()  # Ensure caffeinate is stopped
