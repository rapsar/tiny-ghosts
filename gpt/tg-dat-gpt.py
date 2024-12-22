'''
>> a little bit of an overkill since the same can be achieved by reading the image metadata;
use alternative preferably; still useful proof-of-context; "detail:low" is cheap $0.000213 input <<

Extract Metadata from Images Script

This script processes a folder of images to extract metadata such as date, time, and temperature 
using the GPT-4o API and saves the results to a CSV file. The input folder and output CSV file are 
specified via command-line arguments.

Features:
1. Encodes images as base64 strings and sends them to the GPT-4o API.
2. Requests the extraction of date, time, and temperature (Celsius and Fahrenheit) from each image.
3. Saves the extracted information (filename, date, time, and temperature) to a CSV file.
4. Sorts images alphabetically/numerically for processing.

Usage:
  python tg-dat-gpt.py --input_flash_folder <path_to_flash_folder> --output_csv_name <output_csv_name>
'''

import os
import base64
import csv
import json
import argparse
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to extract date, time, and temperature using OpenAI API with structured output
def extract_image_data(image_path):
    base64_image = encode_image(image_path)

    payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant that extracts date, time, and temperature information from images."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the date, time (yy, mm, dd, hh, mm, ss), and temperature (Celsius and Fahrenheit) from this image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "image_data_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "year": {"type": "string"},
                        "month": {"type": "string"},
                        "day": {"type": "string"},
                        "hour": {"type": "string"},
                        "minute": {"type": "string"},
                        "second": {"type": "string"},
                        "temperature_celsius": {"type": "string"},
                        "temperature_fahrenheit": {"type": "string"}
                    },
                    "required": ["year", "month", "day", "hour", "minute", "second", "temperature_celsius", "temperature_fahrenheit"],
                    "additionalProperties": False
                }
            }
        },
        "max_tokens": 50
    }

    try:
        response = client.chat.completions.create(**payload)
        response_data = response.to_dict()

        # Extract the structured JSON content from the API response
        content = response_data['choices'][0]['message']['content']
        structured_content = json.loads(content)  # Parse the JSON string to a dictionary
        print(f"{structured_content}")

        # Convert the extracted strings to integers or floats where applicable
        image_data = {
            "year": int(structured_content['year']),
            "month": int(structured_content['month']),
            "day": int(structured_content['day']),
            "hour": int(structured_content['hour']),
            "minute": int(structured_content['minute']),
            "second": int(structured_content['second']),
            "temperature_celsius": float(structured_content['temperature_celsius']),
            "temperature_fahrenheit": float(structured_content['temperature_fahrenheit'])
        }

        return image_data

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to append image data to a CSV file
def append_to_csv(file_path, data):
    # Check if file exists
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header if the file is new
        if not file_exists:
            writer.writerow(['Filename', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'TemperatureC', 'TemperatureF'])

        # Write the data row
        writer.writerow(data)

# Main function to process the folder of images
def process_images(input_flash_folder, output_csv_name):
    # Ensure the output CSV is in the same folder as the input folder
    output_csv_path = os.path.join(input_flash_folder, output_csv_name)

    # Sort files in alphabetical/numerical order
    filenames = sorted([f for f in os.listdir(input_flash_folder) if f.lower().endswith(".jpg")])

    for filename in filenames:
        image_path = os.path.join(input_flash_folder, filename)
        print(f"Processing {image_path}...")

        # Extract date, time, and temperature from the image
        extracted_info = extract_image_data(image_path)
        if extracted_info:
            try:
                image_data = [
                    filename,  # Add filename as the first column
                    extracted_info["year"],
                    extracted_info["month"],
                    extracted_info["day"],
                    extracted_info["hour"],
                    extracted_info["minute"],
                    extracted_info["second"],
                    extracted_info["temperature_celsius"],
                    extracted_info["temperature_fahrenheit"]
                ]
                # Append to CSV file
                append_to_csv(output_csv_path, image_data)
            except KeyError:
                print(f"Failed to parse the structured response: {extracted_info}")
        else:
            print(f"Failed to process image: {image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract metadata from images in a folder and save to CSV.")
    parser.add_argument("--input_flash_folder", type=str, required=True, help="Path to the input folder containing flash images.")
    parser.add_argument("--output_csv_name", type=str, required=True, help="Name of the output CSV file (e.g., output.csv).")
    args = parser.parse_args()

    process_images(args.input_flash_folder, args.output_csv_name)
