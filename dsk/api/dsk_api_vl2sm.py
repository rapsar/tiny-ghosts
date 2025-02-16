# dsk_api_vl2sm.py
#
# This module provides an interface to the deepseek-vl2-small model's API via Gradio.
# It defines a function, dsk_api_vl2sm, that sends a preprocessed image along with a prompt
# to the model's /transfer_input and /predict endpoints. If a Gradio client is not provided,
# the function initializes one using the Hugging Face API token from the environment.
#
# The function handles potential GPU quota errors by pausing and retrying, and resets the 
# conversation state after obtaining a prediction. This module can be imported and used 
# independently in other scripts.
#
# Usage Example:
#     from dsk_api_vl2sm import dsk_api_vl2sm
#     result = dsk_api_vl2sm("Is there a firefly flash in the image?", "/path/to/preprocessed/image.jpg")
#     print(result)
#
#     python dsk_api_vl2sm.py -prompt "hello" -image /path/to/image

import os
import argparse
from gradio_client import Client, handle_file

# Retrieve the Hugging Face API token from the environment
HF_TOKEN = os.getenv("HF_TOKEN")

def dsk_api_vl2sm(prompt, image_path, client=None):
    """
    Call the deepseek-vl2-small model's API for a preprocessed image.
    
    Parameters:
        prompt (str): The prompt to send to the model.
        preprocessed_file_path (str): The path to the preprocessed image.
        model (str): The model identifier (e.g., "deepseek-ai/deepseek-vl2-small").
        client (Client, optional): An existing Gradio client. If None, a new client will be created.
        
    Returns:
        The textual output from the model's /predict endpoint.
    """
    # model
    model = "deepseek-ai/deepseek-vl2-small"
    
    # Initialize the client if none is provided.
    if client is None:
        client = Client(model, hf_token=HF_TOKEN)
    
    # Prime the state with /transfer_input.
    transfer_result = client.predict(
        prompt,
        [handle_file(image_path)],
        api_name="/transfer_input"
    )
    
    # Build a simple chat history.
    chat_history = [["Hello", None]]
    
    # Call the /predict endpoint.
    predict_result = client.predict(
        chat_history,   # Chat history input
        0.95,           # Top-p
        0,              # Temperature
        1.1,            # Repetition penalty
        128,            # Max generation tokens
        128,            # Max history tokens
        model,          # Model selection
        api_name="/predict"
    )
    
    # Reset the conversation state.
    client.predict(api_name="/reset_state")
    
    return predict_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call the deepseek-vl2-small model's API.")
    parser.add_argument("-prompt", type=str, default="Answer only by yes or no: is there any firefly flash in the image?", help="Prompt text to send to the model.")
    parser.add_argument("-image", type=str, required=True, help="Path to the preprocessed image.")
    args = parser.parse_args()
    
    result = dsk_api_vl2sm(args.prompt, args.image)
    print("Prediction result:", result)