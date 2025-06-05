import os
import cv2
import numpy as np
import time
import subprocess
import re
import base64
import argparse
import shutil
from PIL import Image, ImageEnhance
from gradio_client import Client, handle_file

# Import the external API call function
from dsk_api_vl2sm import dsk_api_vl2sm
from dsk_api_janus import dsk_api_janus

# Retrieve the Hugging Face API token from the environment
HF_TOKEN = os.getenv("HF_TOKEN")

def preprocess_image(file_path, kernel_diameter, contrast_factor):
    """
    Opens the image, crops the bottom 160 pixels (if possible),
    applies dilation with a disk-shaped structuring element (with given kernel diameter),
    and enhances the contrast.
    
    The processed file is saved to a "temp" folder in the script's directory,
    with '_processed' appended to the original filename.
    
    Returns:
      str: The path to the saved preprocessed image.
    """
    # Open the image
    img = Image.open(file_path).convert("RGB")
    
    # Crop the image
    bottom_banner_height = 160
    width, height = img.size
    if height > bottom_banner_height:
        img = img.crop((0, 0, width, height - bottom_banner_height))
    
    # Convert to NumPy array for dilation
    img_np = np.array(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_diameter, kernel_diameter))
    dilated_np = cv2.dilate(img_np, kernel)
    img = Image.fromarray(dilated_np)
    
    # Enhance the contrast using ImageEnhance
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    
    # Prepare the temp folder path
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    temp_folder = os.path.join(script_dir, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    
    # Build the new filename with '_processed' appended
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    new_filename = f"{name}_processed{ext}"
    temp_file_path = os.path.join(temp_folder, new_filename)
    
    # Save the processed image
    img.save(temp_file_path)
    return temp_file_path

def process_predict_result_janus(predict_result, file_name, original_file_path, output_folder, log_file_path):
    """
    Processes the response from the Janus model's API.
    
    This function performs the following:
      - Prints the result for the given file.
      - Logs the result to a specified log file.
      - If the result (as a string) contains "yes" (case-insensitive), it copies the original image
        to the output folder.
    
    Parameters:
        predict_result (str): The response from the Janus model.
        file_name (str): The name of the processed file.
        original_file_path (str): The full path of the original image.
        output_folder (str): The directory where the image should be copied if positive.
        log_file_path (str): The path to the log file.
    """
    print(f"Result for {file_name}: {predict_result}")

    with open(log_file_path, "a") as log_file:
        log_file.write(f"{file_name}: {predict_result}\n")
    
    if isinstance(predict_result, str) and "yes" in predict_result.lower():
        dest_file_path = os.path.join(output_folder, file_name)
        shutil.copy2(original_file_path, dest_file_path)

def process_predict_result_vl2sm(predict_result, file_name, original_file_path, output_folder, log_file_path):
    """
    Process the predict_result from the API:
      - Extract the text portion (everything before the first <img> tag).
      - Log and print the extracted text.
      - If the text contains "yes" or the full answer contains an <img> tag, copy the original file to the output folder.
      - If an <img> tag is present, extract and save the base64 image using save_extracted_image().
    """
    # Determine the full answer string.
    if isinstance(predict_result, str):
        full_answer = predict_result
    elif isinstance(predict_result, (list, tuple)) and len(predict_result) >= 1:
        chat_history_result = predict_result[0]
        if isinstance(chat_history_result, (list, tuple)) and len(chat_history_result) > 0:
            last_pair = chat_history_result[-1]
            if isinstance(last_pair, (list, tuple)) and len(last_pair) >= 2:
                full_answer = last_pair[1] or ""
            else:
                full_answer = ""
        else:
            full_answer = ""
    else:
        full_answer = ""
    
    # Extract only the text portion (before any <img> tag)
    answer_text = re.split(r'<img.*?>', full_answer)[0]
    
    # Log and print the answer text.
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{file_name}: {answer_text}\n")
    print(f"{file_name}: {answer_text}")
    
    # If the extracted text contains "yes" or an <img> tag is found, copy the original image.
    if "yes" in answer_text.lower() or re.search(r'<img.*?>', full_answer):
        shutil.copy2(original_file_path, os.path.join(output_folder, file_name))
    
    # If there is an <img> tag, extract and save the visual grounding image.
    if re.search(r'<img.*?>', full_answer):
        save_extracted_image(full_answer, output_folder, file_name)

def save_extracted_image(answer_str, output_folder, file_name):
    """
    Look for a base64-encoded image in the answer (inside an <img> tag)
    and save it as a file in the 'vg' subfolder inside the output folder.
    """
    # Use regex to capture the base64 portion inside the <img> tag.
    match = re.search(r'data:image/[^;]+;base64,([^"]+)', answer_str)
    if match:
        try:
            img_bytes = base64.b64decode(match.group(1))
            vg_folder = os.path.join(output_folder, "vg")
            os.makedirs(vg_folder, exist_ok=True)
            dest_file_path = os.path.join(vg_folder, file_name)
            with open(dest_file_path, "wb") as f:
                f.write(img_bytes)
            print(f"Extracted image saved as {dest_file_path}")
            return True
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
    else:
        print("No base64 image found in the answer.")
    return False

def resolve_previous_errors(args, prompt_question):
    log_file_path = os.path.join(args.output, "log.txt")
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as log_file:
            lines = log_file.readlines()
            error_files = []
            for line in lines:
                match = re.match(r"^(\d{8}T\d+_DSCF\d+\.JPG):\s+Error", line)
                if match:
                    error_files.append(match.group(1).strip())

        if error_files:
            with open(log_file_path, "a") as log_file:
                log_file.write("\nReprocessing previously failed files...\n")
                log_file.write("=" * 50 + "\n")

            for file_name in error_files:
                date_folder = file_name.split("T")[0]
                original_file_path = os.path.join(args.input, date_folder, file_name)
                try:
                    preprocessed_file_path = preprocess_image(original_file_path, args.kernel_diameter, args.contrast_factor)
                    print(f"Reprocessing {preprocessed_file_path}...")

                    client = Client("deepseek-ai/Janus-Pro-7B", hf_token=HF_TOKEN)

                    if args.model == "vl2sm":
                        predict_result = dsk_api_vl2sm(prompt_question, preprocessed_file_path, client)
                        process_predict_result_vl2sm(predict_result, file_name, original_file_path, args.output, log_file_path)
                    elif args.model == "janus":
                        predict_result = dsk_api_janus(prompt_question, preprocessed_file_path, client)
                        process_predict_result_janus(predict_result, file_name, original_file_path, args.output, log_file_path)

                except Exception as e:
                    print(f"Retry failed for {file_name}: {e}")
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"{file_name}: Retry Error - {e}\n")
        else:
            print("No error files found to retry.")
    else:
        print(f"No log.txt found in {args.output}. Cannot resolve errors.")

def main(input_folder, output_folder, model_option, kernel_diameter, contrast_factor, prompt_question):
    # Set model_used based on the provided model option.
    if model_option == "janus":
        model_used = "deepseek-ai/Janus-Pro-7B"
    elif model_option == "vl2sm":
        model_used = "deepseek-ai/deepseek-vl2-small"
    else:
        raise ValueError(f"Invalid model option: {model_option}")

    print(f"Using model: {model_used}")
    
    # Now prompt_question is coming in as a parameter rather than being hard-coded.
    print("Using prompt:", prompt_question)
    
    # Ensure the output folder exists.
    os.makedirs(output_folder, exist_ok=True)
    
    # Log file path.
    log_file_path = os.path.join(output_folder, "log.txt")
    processed_files = set()
    
    # Check if log.txt exists. If it does, append a timestamp and parse processed filenames.
    if os.path.exists(log_file_path):
        with open(log_file_path, "a") as log_file:
            log_file.write("\nRun started at " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            log_file.write(f"Input Folder: {input_folder}\n")
            log_file.write("=" * 50 + "\n")
        # Read the file to determine which files have been processed already.
        with open(log_file_path, "r") as log_file:
            for line in log_file:
                # Assuming processed lines are in the format: "filename.jpg: <result>"
                m = re.match(r'^(.+\.jpg):', line, re.IGNORECASE)
                if m:
                    processed_files.add(m.group(1).strip())
    else:
        # Create a new log file with header info.
        with open(log_file_path, "w") as log_file:
            log_file.write(f"Model: {model_used}\n")
            log_file.write(f"Input Folder: {input_folder}\n")
            log_file.write(f"Prompt: {prompt_question}\n")
            log_file.write(f"Kernel Diameter: {kernel_diameter}\n")
            log_file.write(f"Contrast Factor: {contrast_factor}\n")
            #log_file.write(f"Seed: {seed}\n")
            #log_file.write(f"Temperature: {temperature}\n")
            log_file.write("Run started at " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            log_file.write("=" * 50 + "\n")
    
    # Initialize the Gradio client.
    client = Client(model_used, hf_token=HF_TOKEN)
    
    # # Write header info into the log.
    # with open(log_file_path, "w") as log_file:
    #     log_file.write(f"Model: {model_used}\n")
    #     log_file.write(f"Input Folder: {input_folder}\n")
    #     log_file.write(f"Prompt: {prompt_question}\n")
    #     log_file.write(f"Kernel Diameter: {kernel_diameter}\n")
    #     log_file.write(f"Contrast Factor: {contrast_factor}\n")
    #     #log_file.write(f"Seed: {seed}\n")
    #     #log_file.write(f"Temperature: {temperature}\n")
    #     log_file.write("=" * 50 + "\n")
    
    # Process .JPG files in alphabetical order.
    for file_name in sorted(os.listdir(input_folder))[:]:
        if file_name.lower().endswith('.jpg'):
            
            # Skip file if it's already been processed.
            if file_name in processed_files:
                continue
            
            original_file_path = os.path.join(input_folder, file_name)
            # print(f"Processing {original_file_path}...")
            time.sleep(1)
            try:
                # Preprocess the image.
                preprocessed_file_path = preprocess_image(original_file_path, kernel_diameter, contrast_factor)
                print(preprocessed_file_path)
                # Retry loop for handling API errors, up to 5 attempts.
                retry_count = 0
                success = False
                while not success and retry_count < 10:
                    try:
                        if model_option == "vl2sm":
                            # Call the external API function for vl2sm.
                            predict_result = dsk_api_vl2sm(prompt_question, preprocessed_file_path, client)
                            # Process the prediction result.
                            process_predict_result_vl2sm(predict_result, file_name, original_file_path, output_folder, log_file_path)
                        elif model_option == "janus":
                            # Call the external API function for Janus.
                            predict_result = dsk_api_janus(prompt_question, preprocessed_file_path, client) #add client if done with freezing issues, or remove client if persists
                            # Process the Janus-specific result.
                            process_predict_result_janus(predict_result, file_name, original_file_path, output_folder, log_file_path)
                        success = True  # API call succeeded.
                    except Exception as e:
                        retry_count += 1
                        err_msg = str(e)
                        if "exceeded your Pro GPU quota" in err_msg:
                            print("GPU quota exceeded. Sleeping for 1 hour before retrying...")
                            print(time.strftime('%Y-%m-%d %H:%M:%S'))
                            time.sleep(3600)
                        else:
                            print(f"Error on attempt {retry_count} for {file_name}: {e}")
                            time.sleep(2)
                            if retry_count >= 10:
                                raise
                if not success:
                    raise Exception(f"Failed after 10 retries for file {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"{file_name}: Error - {e}\n")

if __name__ == "__main__":
    # Prevent the system from sleeping.
    caffeinate_process = subprocess.Popen(["caffeinate", "-ims"])
    try:
        parser = argparse.ArgumentParser(
            description="Process .JPG files by passing an image and text directly to the deepseek-vl2-small /predict API. "
                        "If the API answer contains 'yes', then the base64-encoded image (if present) is extracted and saved "
                        "into a 'vg' folder within the output folder, and the original image is copied to the output folder."
        )
        parser.add_argument("--input", "-i", required=True, help="Path to the input folder containing .JPG files.")
        parser.add_argument("--output", "-o", required=True, help="Path to the output folder to save images and logs.")
        parser.add_argument("--model", "-m", choices=["janus", "vl2sm"], default="janus", help="Model: deepseek-ai/Janus-Pro-7B or deepseek-ai/deepseek-vl2-small.")
        parser.add_argument("--kernel_diameter", "-kd", type=int, default=5, help="Diameter of the morphological kernel. Default is 5. Set to 1 for no dilatation.")
        parser.add_argument("--contrast_factor", "-cf", type=float, default=1.5, help="Contrast enhancement factor. Default is 1.5. Set to 1 for no modification.")
        parser.add_argument("--vg", action="store_true", help="If specified, wrap 'firefly flash' with visual grounding tags.")
        parser.add_argument("--resolve_errors", action="store_true", help="Retry processing files that previously failed with errors.")
        args = parser.parse_args()
        
        # Define your initial prompt.
        prompt_question = "Answer by yes or no: is there any firefly flash in this image?"
        
        # If the vg flag is provided, apply the tags.
        if args.vg:
            def apply_vg_tags(prompt):
                return re.sub(r"(firefly flash)", r"<|ref|>\1<|/ref|>", prompt, flags=re.IGNORECASE)
            prompt_question = apply_vg_tags(prompt_question)

        # Reprocess files with errors
        if args.resolve_errors:
            resolve_previous_errors(args,prompt_question)
            exit(0)

        main(args.input, args.output, args.model, args.kernel_diameter, args.contrast_factor, prompt_question)
    finally:
        caffeinate_process.terminate()
        
        
    