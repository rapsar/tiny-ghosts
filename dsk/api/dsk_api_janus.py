# dsk_api_janus.py
#
# This module provides an interface to the deepseek-ai/Janus-Pro-7B model's
# multimodal understanding API via Gradio. The function dsk_api_janus sends a
# preprocessed image and a prompt to the model's /multimodal_understanding endpoint.
# If a Gradio client is not provided, it initializes one using the Hugging Face API token.
#
# Usage as a script:
#   python dsk_api_janus.py -prompt "Hello!!" -image /path/to/image
#
# Usage as a module:
#   from dsk_api_janus import dsk_api_janus
#   result = dsk_api_janus("Hello!!", "/path/to/image")
#   print(result)

import os
import argparse
from gradio_client import Client, handle_file

# Retrieve the Hugging Face API token from the environment if needed.
HF_TOKEN = os.getenv("HF_TOKEN")

def dsk_api_janus(prompt, image_path, client=None):
    """
    Call the deepseek-ai/Janus-Pro-7B model's multimodal understanding API.

    Parameters:
        prompt (str): The question or prompt to send to the model.
        image_path (str): The path to the preprocessed image file.
        client (Client, optional): An existing Gradio client instance. If None, a new client is created.

    Returns:
        The prediction result from the model's API.
    """
    model = "deepseek-ai/Janus-Pro-7B"
    if client is None:
        client = Client(model, hf_token=HF_TOKEN)
    
    # Call the API using the provided parameters.
    result = client.predict(
        image=handle_file(image_path),
        question=prompt,
        seed=42,
        top_p=0.95,
        temperature=0,
        api_name="/multimodal_understanding"
    )
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call the deepseek-ai/Janus-Pro-7B model's multimodal understanding API.")
    parser.add_argument("-prompt", type=str, default="Answer only by yes or no: is there any firefly flash in the image?", help="Prompt text to send to the model.")
    parser.add_argument("-image", type=str, required=True, help="Path to the preprocessed image file.")
    args = parser.parse_args()
    
    result = dsk_api_janus(args.prompt, args.image)
    print("Prediction result:", result)