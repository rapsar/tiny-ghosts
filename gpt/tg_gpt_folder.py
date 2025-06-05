'''
Firefly Flash Detection Script

This script processes images for firefly flash detection by splitting them into patches,
classifying them using an AI model, and saving the results in structured logs.
Depending on the patch mode, images are cropped and resized to generate patches, which are then
classified as containing a flash ("yes") or not ("no").

Key changes:
- The `--nonight` flag has been removed; night images are now symlinked into the 'night' folder by default.
- If the input folder contains only subdirectories, each subfolder is processed separately into matching output subfolders.

Features:
- Supports multiple patch modes: full image, two-patch split, bottom/top crops, or eight-patch split.
- Saves classification results in both JSON and text log files.
- Organizes images into 'flash' (copied), 'night' (symlinked), 'temp', and 'positive-patches'.

Usage:
  python tg_gpt_folder.py --input <input_folder> --output <output_folder> \
         [--model <model_name>] [--patch <patch_mode>] [--kernel_diameter N] \
         [--contrast_factor F] [--detail <low|high>] [--sensitivity S]
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
import time

# Import the new preprocess_image function from the separate module.
from tg_gpt_preprocess_image import preprocess_image

# Initialize OpenAI client
client = OpenAI()

def encode_image(image_path):
    """Encodes the image from a file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def classify_image(image_path, model="gpt-4o", detail="high"):
    """
    Classifies the image using the OpenAI API.
    Returns the response text and top token probabilities.
    """
    prompt = "Answer only by yes or no: do you see any firefly flashes in this image? (watch very carefully)"
    
    base64_image = encode_image(image_path)
    
    temperature=0
    top_p=0.1
    logprobs=True
    top_logprobs=2

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": detail
                                }
                        }
                    ]
                }
            ]
        )

        # Extract the response text
        response_text = response.choices[0].message.content.strip().lower()

        # Extract token information
        logprobs_content = response.choices[0].logprobs.content
        top_token_info = []
        for token_logprob in logprobs_content[:1]:
            for top in token_logprob.top_logprobs[:2]:
                token = top.token
                logprob = top.logprob
                probability = math.exp(logprob)  # Convert log probability to probability
                top_token_info.append((token, probability))

        return response_text, top_token_info
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "error", []

def process_images(input_folder, output_folder, model, patch_mode, kernel_diameter, contrast_factor, detail, sensitivity, master_input=None):
    """
    Processes all JPG images in the input folder.
    For each image, the new preprocess_image function (which returns PIL.Image patches) is called.
    Each patch is temporarily saved and classified. Results are logged and images are organized
    based on the classification output.
    """
    flash_folder = os.path.join(output_folder, 'flash')
    temp_folder = os.path.join(output_folder, 'temp')
    positive_patches_folder = os.path.join(output_folder, 'positive-patches')
    json_log_path = os.path.join(output_folder, f"results_{os.path.basename(input_folder)}.json")

    # Determine night-folder, placing subfolder symlinks under output/night/<subfolder>
    if master_input:
        rel = os.path.relpath(input_folder, master_input)
        if rel == '.':
            night_folder = os.path.join(output_folder, 'night')
        else:
            night_folder = os.path.join(output_folder, 'night', rel)
    else:
        night_folder = os.path.join(output_folder, 'night')

    os.makedirs(flash_folder, exist_ok=True)
    os.makedirs(night_folder, exist_ok=True)
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(positive_patches_folder, exist_ok=True)

    # Initialize the JSON log file if it doesn't exist.
    if not os.path.exists(json_log_path):
        with open(json_log_path, "w") as json_file:
            json.dump({"images": []}, json_file)

    input_folder_name = os.path.basename(input_folder)
    log_file_path = os.path.join(output_folder, f"results_{input_folder_name}.txt")

    # Append a run header with timestamp to avoid overwriting existing logs
    with open(log_file_path, "a") as log_file:
        log_file.write("\nRun started at " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        log_file.write(f"Model: {model}\n")
        log_file.write(f"Detail: {detail}\n")
        log_file.write(f"Sensitivity: {sensitivity}\n")
        log_file.write(f"Kernel Diameter: {kernel_diameter}\n")
        log_file.write(f"Contrast Factor: {contrast_factor}\n")
        log_file.write("=" * 50 + "\n\n")

    with open(log_file_path, "a") as log_file:
        # Process JPG files that match the naming pattern.
        filenames = sorted([f for f in os.listdir(input_folder)
                            if f.lower().endswith('.jpg') ]) #and f.startswith('20')])
        # Skip images already processed (in flash or night folders)
        processed = set()
        # Images copied to flash
        for f in os.listdir(flash_folder):
            processed.add(f)
        # Images symlinked to night
        for f in os.listdir(night_folder):
            processed.add(f)
        # Only process new images
        filenames = [f for f in filenames if f not in processed]
        for filename in filenames:
            image_path = os.path.join(input_folder, filename)
            try:
                # Preprocess the image to obtain patches (returned as PIL.Image objects).
                patches = preprocess_image(file_path=image_path, patch_mode=patch_mode,
                                           kernel_diameter=kernel_diameter,
                                           contrast_factor=contrast_factor)
                destination_folder = night_folder
                image_entry = {"image_path": image_path, "patches": []}

                positive_found = False
                for idx, patch in enumerate(patches):
                    # Save each patch temporarily in the temp folder.
                    temp_patch_path = os.path.join(temp_folder, f"temp_patch_{idx}.jpg")
                    patch.save(temp_patch_path)

                    result, top_token_info = classify_image(temp_patch_path, model=model, detail=detail)

                    log_file.write(f"Image: {image_path}\nPatch: {temp_patch_path}\nOutput: {result}\n")
                    print(f"Image: {image_path}")
                    print(f"Patch: {temp_patch_path}")
                    print(f"Output: {result}")

                    patch_entry = {
                        "patch_path": temp_patch_path,
                        "output": result,
                        "tokens": [{"token": token, "probability": probability} 
                                   for token, probability in top_token_info]
                    }
                    image_entry["patches"].append(patch_entry)

                    for token, probability in top_token_info:
                        print(f"Token: '{token}', Probability: {probability:.2f}")
                        log_file.write(f"Token: '{token}', Probability: {probability:.2f}\n")
                    log_file.write("\n")
                    print("-----------------------------")
                    
                    # Check if any token labeled 'yes' exceeds the sensitivity threshold.
                    for token, probability in top_token_info:
                        if token.lower() == "yes" and probability > sensitivity:
                            positive_found = True
                            break

                    if 'yes' in result or positive_found:
                        # If positive, copy the patch to the positive patches folder.
                        patch_filename = os.path.basename(temp_patch_path)
                        positive_patch_path = os.path.join(positive_patches_folder, f"{filename}_{patch_filename}")
                        shutil.copy(temp_patch_path, positive_patch_path)
                        print(f"Saved positive patch: {positive_patch_path}")
                        destination_folder = flash_folder
                        break

                    # Remove the temporary patch file after classification.
                    os.remove(temp_patch_path)

                # Append image entry to JSON log.
                with open(json_log_path, "r+") as json_file:
                    data = json.load(json_file)
                    data["images"].append(image_entry)
                    json_file.seek(0)
                    json.dump(data, json_file, indent=4)

                # Copy or symlink the original image to the appropriate folder based on classification.
                try:
                    if destination_folder == night_folder:
                        # Use symlink for night images.
                        symlink_path = os.path.join(destination_folder, filename)
                        if not os.path.exists(symlink_path):
                            os.symlink(os.path.abspath(image_path), symlink_path)
                        print(f"Symlinked {filename} to {destination_folder}")
                    else:
                        shutil.copy(image_path, destination_folder)
                        print(f"Copied {filename} to {destination_folder}")
                except Exception as e:
                    print(f"Error copying/symlinking image {filename}: {e}")
                    log_file.write(f"Image: {image_path}\nError copying/symlinking file: {e}\n\n")
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                log_file.write(f"Image: {image_path}\nError: {e}\n\n")

    # Clean up temp folder when done
    try:
        shutil.rmtree(temp_folder)
        print(f"Removed temp folder: {temp_folder}")
    except Exception as e:
        print(f"Error removing temp folder {temp_folder}: {e}")

if __name__ == "__main__":
    # Prevent system sleep.
    caffeinate_process = subprocess.Popen(["caffeinate", "-ims"])
    try:
        parser = argparse.ArgumentParser(description="Process images for firefly flash detection.")
        parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input folder containing images.")
        parser.add_argument("--output", "-o", type=str, required=True, help="Path to the output folder for classified images.")
        parser.add_argument("--model", "-m", type=str, default="gpt-4o", choices=[
            "gpt-4o",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano"
        ], help="Model to use for classification (default: gpt-4o).")
        parser.add_argument("--patch", type=str, default="2", choices=["1", "2", "4d", "4u", "8"],
                            help="Patch mode: '1' (full image), '2' (two 512x512 patches), "
                                 "'4d' (four 512x512 patches from resized bottom crop), "
                                 "'4u' (four 512x512 patches from resized top crop), "
                                 "or '8' (eight 512x512 patches). Default is '2'.")
        # --nonight removed
        parser.add_argument("--kernel_diameter", "-kd", type=int, default=3, help="Kernel diameter for dilation (default: 3).")
        parser.add_argument("--contrast_factor", "-cf", type=float, default=1.5, help="Contrast enhancement factor (default: 1.5).")
        parser.add_argument("--detail", type=str, default="high", choices=["low", "high"], help="Pass image as low (85 tokens) or high detail.")
        parser.add_argument("--sensitivity", type=float, default=0.5, help="Probability threshold for classifying a patch as positive (default: 0.5).")
        args = parser.parse_args()

        # If no JPEG files at top level, process each subfolder separately
        top_jpg = [f for f in os.listdir(args.input) if f.lower().endswith('.jpg')]
        if not top_jpg:
            # Determine global flash folder for skip logic
            flash_folder = os.path.join(args.output, 'flash')
            for sub in sorted(os.listdir(args.input)):
                sub_input = os.path.join(args.input, sub)
                if os.path.isdir(sub_input):
                    # Skip entire subfolder if all images already processed
                    input_files = [f for f in os.listdir(sub_input) if f.lower().endswith('.jpg')]
                    night_subfolder = os.path.join(args.output, 'night', sub)
                    existing = set()
                    if os.path.exists(night_subfolder):
                        existing.update(os.listdir(night_subfolder))
                    if os.path.exists(flash_folder):
                        existing.update([f for f in os.listdir(flash_folder) if f in input_files])
                    if set(input_files).issubset(existing):
                        print(f"Skipping already processed folder: {sub_input}")
                        continue
                    process_images(
                        sub_input, args.output, args.model, args.patch,
                        kernel_diameter=args.kernel_diameter,
                        contrast_factor=args.contrast_factor,
                        detail=args.detail,
                        sensitivity=args.sensitivity,
                        master_input=args.input
                    )
        else:
            process_images(
                args.input, args.output, args.model, args.patch,
                kernel_diameter=args.kernel_diameter,
                contrast_factor=args.contrast_factor,
                detail=args.detail,
                sensitivity=args.sensitivity,
                master_input=args.input
            )
    finally:
        caffeinate_process.terminate()  # Stop caffeinate when done.
        
# python tg_gpt_folder.py --model "gpt-4.1-mini" --input /Users/rss367/Desktop/2024bww/Muleshoe/results/UpperBass/dusk --output /Users/rss367/Desktop/2024bww/Muleshoe/results/UpperBass/dusk/_41mini