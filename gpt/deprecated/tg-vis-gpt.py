'''
Firefly Flash Detection Script

This script processes images for firefly flash detection by splitting them into patches, classifying them using an AI model, and saving the results in structured logs. 
Depending on the patch mode, images are cropped and resized to generate patches, which are then classified as either containing a flash ('yes') or not ('no').

Features:
- Supports multiple patch modes: full image, two-patch split, bottom crop with resizing, or eight-patch split.
- Saves classification results in JSON and text log files.
- Organizes images into folders based on classification (e.g., flash, night).
- Optionally prevents night images from being copied to the output folder.

Usage:
  python script.py --input <input_folder> --output <output_folder> --patch <patch_mode> --model <model_name>
'''

import os
import shutil
import base64
import json
import argparse
from PIL import Image
from openai import OpenAI
import math
import subprocess

# Initialize OpenAI client
client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to classify the image using OpenAI API
def classify_image(image_path, model="gpt-4o"):
    base64_image = encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.01,
            top_p=0.1,
            logprobs=True,
            top_logprobs=2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Answer only by yes or no: do you see any firefly flashes in this image? (watch very carefully)"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]
        )

        # Extract the response text
        response_text = response.choices[0].message.content.strip().lower()

        # Display top 2 tokens and probabilities for the first token of the output
        logprobs_content = response.choices[0].logprobs.content
        top_token_info = []
        for idx, token_logprob in enumerate(logprobs_content[:1], start=1):
            top_two = token_logprob.top_logprobs[:2]
            for top in top_two:
                token = top.token
                logprob = top.logprob
                probability = math.exp(logprob)  # Convert logprob to probability
                top_token_info.append((token, probability))

        return response_text, top_token_info
    except Exception as e:
        # Handle errors from OpenAI API
        print(f"OpenAI API error: {e}")
        return "error", []

# Function to crop the image based on the --patch argument
def crop_image(image_path, temp_dir, patch_mode):
    with Image.open(image_path) as img:
        width, height = img.size

        # Calculate the top margin to achieve a final height of 1280
        crop_height = 1280
        bottom_margin = 140
        top_margin = height - crop_height - bottom_margin

        # Crop the top and bottom to get 1280x2560
        cropped_img = img.crop((0, top_margin, width, height - bottom_margin))

        if patch_mode == "1":
            # Save the full image
            full_patch_path = os.path.join(temp_dir, "_full-patch.JPG")
            cropped_img.save(full_patch_path)
            return [full_patch_path]

        elif patch_mode == "2":
            # Resize to 512x1024 and split into two 512x512 patches
            resized_img = cropped_img.resize((1024, 512), Image.LANCZOS)

            left_patch = resized_img.crop((0, 0, 512, 512))
            right_patch = resized_img.crop((512, 0, 1024, 512))

            left_patch_path = os.path.join(temp_dir, "_left-patch.JPG")
            right_patch_path = os.path.join(temp_dir, "_right-patch.JPG")

            left_patch.save(left_patch_path)
            right_patch.save(right_patch_path)

            return [left_patch_path, right_patch_path]

        elif patch_mode == "4d":
            # Crop the top 640 pixels, keeping only the bottom 640x2560
            bottom_cropped_img = cropped_img.crop((0, 640, width, 1280))

            # Resize to 512x2048 (factor 0.8)
            resized_img = bottom_cropped_img.resize((2048, 512), Image.LANCZOS)

            # Split into four 512x512 patches
            patch_paths = []
            for i in range(4):
                patch = resized_img.crop((i * 512, 0, (i + 1) * 512, 512))
                patch_path = os.path.join(temp_dir, f"_patch_4_{i}.JPG")
                patch.save(patch_path)
                patch_paths.append(patch_path)

            return patch_paths
        
        elif patch_mode == "4u":
            # Crop the top 640 pixels
            top_cropped_img = cropped_img.crop((0, 0, width, 640))

            # Resize to 512x2048 (factor 0.8)
            resized_img = top_cropped_img.resize((2048, 512), Image.LANCZOS)

            # Split into four 512x512 patches
            patch_paths = []
            for i in range(4):
                patch = resized_img.crop((i * 512, 0, (i + 1) * 512, 512))
                patch_path = os.path.join(temp_dir, f"_patch_4_{i}.JPG")
                patch.save(patch_path)
                patch_paths.append(patch_path)

            return patch_paths

        elif patch_mode == "8":
            # Resize to 1024x2048 and split into eight 512x512 patches
            resized_img = cropped_img.resize((2048, 1024), Image.LANCZOS)

            patch_paths = []
            for i in range(1, -1, -1):
                for j in range(4):
                    patch = resized_img.crop((j * 512, i * 512, (j + 1) * 512, (i + 1) * 512))
                    patch_path = os.path.join(temp_dir, f"_patch_{i}_{j}.JPG")
                    patch.save(patch_path)
                    patch_paths.append(patch_path)

            return patch_paths

        else:
            raise ValueError("Invalid patch mode. Must be 1, 2, 4u, 4d, or 8.")

# Function to process images in a folder
def process_images(input_folder, output_folder, model, patch_mode, nonight):
    # Create output folders if they don't exist
    flash_folder = os.path.join(output_folder, 'flash')
    night_folder = os.path.join(output_folder, 'night')
    temp_folder = os.path.join(output_folder, 'temp')
    positive_patches_folder = os.path.join(output_folder, 'positive-patches')
    json_log_path = os.path.join(output_folder, f"results_{os.path.basename(input_folder)}.json")

    os.makedirs(flash_folder, exist_ok=True)
    if not nonight:
        os.makedirs(night_folder, exist_ok=True)
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(positive_patches_folder, exist_ok=True)

    # Ensure the JSON log file exists
    if not os.path.exists(json_log_path):
        with open(json_log_path, "w") as json_file:
            json.dump({"images": []}, json_file)

    # Get the name of the input folder
    input_folder_name = os.path.basename(input_folder)
    log_file_path = os.path.join(output_folder, f"results_{input_folder_name}.txt")

    # Open the log file for writing
    with open(log_file_path, "w") as log_file:
        # Get list of JPG files and sort them
        filenames = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.jpg') and f.startswith('DSCF')])

        # Iterate over sorted list of JPG files
        for filename in filenames:
            image_path = os.path.join(input_folder, filename)

            try:
                # Use the temp folder for patch files
                patch_paths = crop_image(image_path, temp_folder, patch_mode)

                destination_folder = night_folder if not nonight else None
                image_entry = {"image_path": image_path, "patches": []}

                for patch_path in patch_paths:
                    result, top_token_info = classify_image(patch_path, model=model)

                    # Log in text file
                    log_file.write(f"Image: {image_path}\nPatch: {patch_path}\nOutput: {result}\n")
                    print(f"Image: {image_path}")
                    print(f"Patch: {patch_path}")
                    print(f"Output: {result}")

                    # Build JSON patch entry
                    patch_entry = {
                        "patch_path": patch_path,
                        "output": result,
                        "tokens": [{"token": token, "probability": probability} for token, probability in top_token_info]
                    }

                    image_entry["patches"].append(patch_entry)

                    # Log tokens in text file
                    for token, probability in top_token_info:
                        print(f"Token: '{token}', Probability: {probability:.6f}")
                        log_file.write(f"Token: '{token}', Probability: {probability:.6f}\n")

                    log_file.write("\n")
                    print("-----------------------------")  # Separator for console output

                    if 'yes' in result:
                        # Save the positive patch
                        patch_filename = os.path.basename(patch_path)
                        positive_patch_path = os.path.join(positive_patches_folder, f"{filename}_{patch_filename}")
                        shutil.copy(patch_path, positive_patch_path)
                        print(f"Saved positive patch: {positive_patch_path}")

                        destination_folder = flash_folder
                        break

                # Append image entry to JSON log
                with open(json_log_path, "r+") as json_file:
                    data = json.load(json_file)
                    data["images"].append(image_entry)
                    json_file.seek(0)
                    json.dump(data, json_file, indent=4)

                # Copy the original image to the appropriate folder
                if destination_folder:
                    try:
                        shutil.copy(image_path, destination_folder)
                        print(f"Copied {filename} to {destination_folder}")
                    except Exception as e:
                        print(f"Error copying image {filename}: {e}")
                        log_file.write(f"Image: {image_path}\nError copying file: {e}\n\n")
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                log_file.write(f"Image: {image_path}\nError: {e}\n\n")

if __name__ == "__main__":
    caffeinate_process = subprocess.Popen(["caffeinate", "-ims"])  # Start caffeinate
    try:
        parser = argparse.ArgumentParser(description="Process images for firefly flash detection.")
        parser.add_argument("--input", type=str, required=True, help="Path to the input folder containing images.")
        parser.add_argument("--output", type=str, required=True, help="Path to the output folder for classified images.")
        parser.add_argument("--model", type=str, default="gpt-4o", choices=[
            "gpt-4o",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13",
            "gpt-4o-mini"
        ], help="Model to use for classification (default: gpt-4o).")
        #parser.add_argument("--patch", type=int, default=2, choices=[1, 2, 4, 8], help="Patch mode: 1 (full image), 2 (two 512x512 patches), 4 (four 512x512 patches from resized bottom crop), or 8 (eight 512x512 patches). Default is 2.")
        parser.add_argument("--patch", type=str, default="2", choices=["1", "2", "4d", "4u", "8"],
                    help="Patch mode: 1 (full image), 2 (two 512x512 patches), "
                         "4d (four 512x512 patches from resized bottom crop), "
                         "4u (four 512x512 patches from resized top crop), "
                         "or 8 (eight 512x512 patches). Default is 2.")
        parser.add_argument("--nonight", action="store_true", help="Do not copy images classified as 'night' to the output folder.")
        args = parser.parse_args()

        process_images(args.input, args.output, args.model, args.patch, args.nonight)
    finally:
        caffeinate_process.terminate()  # Ensure caffeinate is stopped

