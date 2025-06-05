"""
Firefly Flash Detection - Single Image Test Script

This script allows testing firefly flash detection on a single image.
It preprocesses the image using a separate module (tg_gpt_preprocess_image.py)
that crops, enhances, and splits the image into patches (returned as PIL.Image objects).
Each patch is then temporarily saved and classified using an AI model.
Classification results and token probabilities are output directly to the console.

Features:
- Supports multiple patch modes: full image, two-patch split, bottom crop with resizing, four-patch (top or bottom), or eight-patch split.
- Outputs classification results and token probabilities for each patch.
- Uses a modular preprocessing function from tg_gpt_preprocess_image.py.

Usage:
  python tg-vis-gpt-test.py --input <image_path> --patch <patch_mode> --model <model_name>
"""

import os
import base64
import argparse
from PIL import Image
from openai import OpenAI
import math

# Import the new preprocess_image function from the separate module.
from tg_gpt_preprocess_image import preprocess_image

# Initialize OpenAI client
client = OpenAI()

def encode_image(image_path):
    """
    Encodes the image from a file to a base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def classify_image(image_path, model="gpt-4o"):
    """
    Classifies the image using the OpenAI API.
    Returns the response text and top token probabilities.
    """
    prompt = "Answer only by yes or no: do you see any firefly flashes in this image? (watch very carefully)"
    
    base64_image = encode_image(image_path)
    
    detail = "low"
    
    temperature=0
    seed = 39 # non effective (for now 02/2025 -- might change in the future)
    top_p=0.1
    logprobs=True
    top_logprobs=2

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            messages=[
                    {
        "role": "system",
        "content": ""
                    },
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
        print(f"prompt_tokens: {response.usage.prompt_tokens}")
        print(f"system_fingerprint: {response.system_fingerprint}")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a single test image for firefly flash detection.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model", type=str, default="gpt-4o", choices=[
        "gpt-4o",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini",
        "gpt-4.5-preview",
        #"o1-mini",
        #"o3-mini"
    ], help="Model to use for classification (default: gpt-4o).")
    parser.add_argument("--patch", type=str, default="2", choices=["1", "2", "4d", "4u", "8"],
                        help="Patch mode: '1' (full image), '2' (two 512x512 patches), '4d' (four 512x512 patches from bottom crop), '4u' (four 512x512 patches from top crop), or '8' (eight 512x512 patches). Default is '2'.")
    parser.add_argument("--kernel_diameter", "-kd", type=int, default=1, help="Kernel diameter for dilation (default: 1).")
    parser.add_argument("--contrast_factor", "-cf", type=float, default=1, help="Contrast enhancement factor (default: 1).")
    args = parser.parse_args()

    # Preprocess the image to obtain patches (as PIL.Image objects)
    patches = preprocess_image(
        file_path=args.input,
        patch_mode=args.patch,
        kernel_diameter=args.kernel_diameter,
        contrast_factor=args.contrast_factor
    )

    # Process each patch: save it temporarily, classify it, print the results, and remove the temporary file.
    for idx, patch in enumerate(patches):
        temp_patch_path = f"temp_patch_{idx}.jpg"
        patch.save(temp_patch_path)
        patch.show()
        result, top_token_info = classify_image(temp_patch_path, model=args.model)

        print(f"Patch {idx + 1}:")
        print(f"Output: {result}")
        for token, probability in top_token_info:
            print(f"Token: '{token}', Probability: {probability:.2f}")
        print("-----------------------------")

        # Remove the temporary patch file
        os.remove(temp_patch_path)