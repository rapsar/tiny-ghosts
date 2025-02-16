"""
tg_gpt_preprocess_image.py

This module provides a preprocessing pipeline for images to be passed to the ChatGPT API.
It performs the following operations:
  1. Crops the image to a fixed height with a defined bottom margin.
  2. Enhances the image by applying dilation and contrast enhancement.
  3. Splits the enhanced image into patches based on a specified patch mode.

Supported Patch Modes:
    "1"  - Returns the full image as a single patch.
    "2"  - Resizes to 1024x512 and splits into two 512x512 patches (default).
    "4d" - Crops the bottom half (from 640 to 1280), resizes to 2048x512, then splits into four patches.
    "4u" - Crops the top half (first 640 pixels), resizes to 2048x512, then splits into four patches.
    "8"  - Resizes to 2048x1024 and splits into eight 512x512 patches.

Usage Example:
    from tg_gpt_preprocess_image import preprocess_image
    patches = preprocess_image("path/to/image.jpg", kernel_diameter=5, contrast_factor=1.5, patch_mode="2")
    # 'patches' is a list of PIL.Image objects ready for further processing.

Date: 02/2025
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance

def crop_image(image_path):
    """
    Opens the image and crops it using preset parameters.

    The image is cropped to a fixed height (1280) with a bottom margin (140).

    Returns:
        PIL.Image: The cropped image.
    """
    with Image.open(image_path) as img:
        width, height = img.size
        crop_height = 1280
        bottom_margin = 140
        top_margin = height - crop_height - bottom_margin

        # Crop the image: (left, top, right, bottom)
        cropped_img = img.crop((0, top_margin, width, height - bottom_margin))
        return cropped_img

def enhance_image(image, kernel_diameter, contrast_factor):
    """
    Enhances the image by applying dilation and contrast enhancement.

    Parameters:
        image (PIL.Image): The image to enhance.
        kernel_diameter (int): Diameter of the morphological kernel.
        contrast_factor (float): Factor to enhance contrast.

    Returns:
        PIL.Image: The enhanced image.
    """
    # Convert image to NumPy array for dilation
    img_np = np.array(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_diameter, kernel_diameter))
    dilated_np = cv2.dilate(img_np, kernel)
    img_dilated = Image.fromarray(dilated_np)
    
    # Enhance the contrast
    enhancer = ImageEnhance.Contrast(img_dilated)
    enhanced_img = enhancer.enhance(contrast_factor)
    
    return enhanced_img

def split_into_patches(image, patch_mode):
    """
    Splits the enhanced image into patches based on the specified patch mode.
    
    Supported patch modes:
        "1"  - Returns the full image as a single patch.
        "2"  - Resizes to 1024x512 and splits into two 512x512 patches.
        "4d" - Crops the bottom half (from 640 to 1280), resizes to 2048x512, then splits into four patches.
        "4u" - Crops the top half (first 640 pixels), resizes to 2048x512, then splits into four patches.
        "8"  - Resizes to 2048x1024 and splits into eight 512x512 patches.
    
    Returns:
        list of PIL.Image: A list of image patches.
    """
    width, height = image.size

    if patch_mode == "1":
        return [image]

    elif patch_mode == "2":
        # Resize to 1024x512 and split into two 512x512 patches
        resized_img = image.resize((1024, 512), Image.LANCZOS)
        left_patch = resized_img.crop((0, 0, 512, 512))
        right_patch = resized_img.crop((512, 0, 1024, 512))
        return [left_patch, right_patch]

    elif patch_mode == "4d":
        # Crop the bottom portion: from 640 to 1280
        bottom_cropped_img = image.crop((0, 640, width, 1280))
        resized_img = bottom_cropped_img.resize((2048, 512), Image.LANCZOS)
        patches = []
        for i in range(4):
            patch = resized_img.crop((i * 512, 0, (i + 1) * 512, 512))
            patches.append(patch)
        return patches

    elif patch_mode == "4u":
        # Crop the top portion: first 640 pixels
        top_cropped_img = image.crop((0, 0, width, 640))
        resized_img = top_cropped_img.resize((2048, 512), Image.LANCZOS)
        patches = []
        for i in range(4):
            patch = resized_img.crop((i * 512, 0, (i + 1) * 512, 512))
            patches.append(patch)
        return patches

    elif patch_mode == "8":
        # Resize to 2048x1024 and split into eight 512x512 patches
        resized_img = image.resize((2048, 1024), Image.LANCZOS)
        patches = []
        for i in range(1, -1, -1):
            for j in range(4):
                patch = resized_img.crop((j * 512, i * 512, (j + 1) * 512, (i + 1) * 512))
                patches.append(patch)
        return patches

    else:
        raise ValueError("Invalid patch mode. Must be one of: '1', '2', '4d', '4u', or '8'.")

def preprocess_image(file_path, kernel_diameter=5, contrast_factor=1.5, patch_mode="2"):
    """
    Full preprocessing pipeline:
      1. Crop the image.
      2. Enhance the image (dilation + contrast enhancement).
      3. Split the image into patches based on the selected mode.
    
    Parameters:
        file_path (str): Path to the input image.
        patch_mode (str): One of "1", "2", "4d", "4u", or "8". Default is "2".
        kernel_diameter (int): Diameter for dilation (default: 5).
        contrast_factor (float): Factor for contrast enhancement (default: 1.5).
    
    Returns:
        list of PIL.Image: A list of processed image patches.
    """
    cropped = crop_image(file_path)
    enhanced = enhance_image(cropped, kernel_diameter, contrast_factor)
    patches = split_into_patches(enhanced, patch_mode)
    return patches