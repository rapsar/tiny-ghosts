'''
Firefly Flash Detection - Single Image Test Script

This script allows testing firefly flash detection on a single image. It crops the image
into patches based on the selected patch mode, classifies the patches using an AI model,
and outputs the classification results directly to the console.

Features:
- Supports multiple patch modes: full image, two-patch split, bottom crop with resizing, or eight-patch split.
- Outputs classification results and token probabilities for each patch.
- Suitable for testing and debugging model performance on individual images.

Usage:
  python test_image_classification.py --input <image_path> --patch <patch_mode> --model <model_name>
'''

import os
import base64
import argparse
from PIL import Image
from openai import OpenAI
import math
from PIL import ImageFilter

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
def crop_image(image_path, patch_mode):
    with Image.open(image_path) as img:
        width, height = img.size

        # Calculate the top margin to achieve a final height of 1280
        crop_height = 1280
        bottom_margin = 140
        top_margin = height - crop_height - bottom_margin

        # Crop the top and bottom to get 1280x2560
        cropped_img = img.crop((0, top_margin, width, height - bottom_margin))

        if patch_mode == 1:
            # Return the full image
            return [cropped_img]

        elif patch_mode == 2:
            # Resize to 512x1024 and split into two 512x512 patches
            resized_img = cropped_img.resize((1024, 512), Image.LANCZOS)
            #resized_img = cropped_img.resize((1024, 512), Image.LANCZOS).filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=0))

            left_patch = resized_img.crop((0, 0, 512, 512))
            right_patch = resized_img.crop((512, 0, 1024, 512))

            return [left_patch, right_patch]
        
        elif patch_mode == 4:
            # Crop the top 660 pixels, keeping only the bottom 640x2560
            bottom_cropped_img = cropped_img.crop((0, 660, width, 1280))

            # Resize to 512x2048 (factor 0.8)
            resized_img = bottom_cropped_img.resize((2048, 512), Image.LANCZOS)

            # Split into four 512x512 patches
            patches = []
            for i in range(4):
                patch = resized_img.crop((i * 512, 0, (i + 1) * 512, 512))
                patches.append(patch)

            return patches

        elif patch_mode == 8:
            # Resize to 1024x2048 and split into eight 512x512 patches
            resized_img = cropped_img.resize((2048, 1024), Image.LANCZOS)

            patches = []
            for i in range(1, -1, -1):
                for j in range(4):
                    patch = resized_img.crop((j * 512, i * 512, (j + 1) * 512, (i + 1) * 512))
                    patches.append(patch)

            return patches

        else:
            raise ValueError("Invalid patch mode. Must be 1, 2, or 8.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a single test image for firefly flash detection.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model", type=str, default="gpt-4o", choices=[
        "gpt-4o",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini"
    ], help="Model to use for classification (default: gpt-4o).")
    parser.add_argument("--patch", type=int, default=2, choices=[1, 2, 4, 8], help="Patch mode: 1 (full image), 2 (two 512x512 patches), 4 (four 512x512 patches from resized bottom crop), or 8 (eight 512x512 patches). Default is 2.")
    args = parser.parse_args()

    # Crop the image into patches based on patch_mode
    patches = crop_image(args.input, args.patch)

    for idx, patch in enumerate(patches):
        # Save patch to temporary memory and classify
        patch_path = f"temp_patch_{idx}.jpg"
        patch.save(patch_path)
        result, top_token_info = classify_image(patch_path, model=args.model)

        print(f"Patch {idx + 1}:")
        print(f"Output: {result}")
        for token, probability in top_token_info:
            print(f"Token: '{token}', Probability: {probability:.6f}")
        print("-----------------------------")

        # Remove the temporary patch file
        os.remove(patch_path)
