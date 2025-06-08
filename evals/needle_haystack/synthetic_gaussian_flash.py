
import base64
from openai import OpenAI
import math

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Initialize OpenAI client
client = OpenAI()

def encode_image(image_path):
    """Encodes the image from a file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def classify_image(image_path, model, detail="high"):
    """
    Classifies the image using the OpenAI API.
    Returns the response text and top token probabilities.
    """
    prompt = "Answer only by yes or no: do you see any firefly flashes in this image? (watch very carefully)"
    
    base64_image = encode_image(image_path)
    
    seed = 0
    temperature=0
    top_p=0.1
    logprobs=True
    top_logprobs=4

    try:
        response = client.chat.completions.create(
            model=model,
            seed = seed,
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
            for top in token_logprob.top_logprobs:
                token = top.token
                logprob = top.logprob
                probability = math.exp(logprob)  # Convert log probability to probability
                top_token_info.append((token, probability))

        return response_text, top_token_info
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "error", []

def add_gaussian_spot(img, x, y, sigma, peak):
    """
    Add a white Gaussian “spot” to img at position (x,y).
    img: HxWx3 uint8 array
    sigma: standard deviation of the Gaussian
    peak: maximum value at the center
    """
    # if want a simple circle instead
    # cv2.circle(img, (x, y), sigma, (peak, peak, peak), thickness=-1)
    
    kernel_radius = 16
    kernel_size = 2*kernel_radius + 1
    c = cv2.getGaussianKernel(kernel_size, sigma)
    cc = c.dot(c.T)
    ccn = cc / np.max(cc)
    cccc = peak * ccn

    half = kernel_radius

    # region of interest in the image
    x0, x1 = max(0, x-half), min(w, x+half+1)
    y0, y1 = max(0, y-half), min(h, y+half+1)

    # corresponding region of the kernel
    kx0, kx1 = half - (x - x0), half + (x1 - x)
    ky0, ky1 = half - (y - y0), half + (y1 - y)

    # add (with clipping) to all three channels
    spot = cccc[ky0:ky1, kx0:kx1].astype(np.uint8)
    for c in range(3):
        img[y0:y1, x0:x1, c] = np.clip(
        img[y0:y1, x0:x1, c].astype(int) + spot.astype(int),
        0, 255
    ).astype(np.uint8)


# Sweep a white blob over the image and record 'Yes' probability
h, w = 512, 1024
fill_value = 8
step = 16
xs = list(range(0, w, step))
ys = list(range(0, h, step))
# xs = list(range(192, 216, 1))
# ys = list(range(280, 288, 1))
# Store x, y, and yes probability in a 3D array
result_array = np.zeros((len(ys), len(xs), 3), dtype=float)

model = "gpt-4.1-mini"
peak = 64
sigma = 3

for iy, y in enumerate(tqdm(ys, desc="Y positions")):
    for ix, x in enumerate(tqdm(xs, desc="X positions", leave=False)):
        # create base image
        img = np.full((h, w, 3), fill_value=fill_value, dtype=np.uint8)
        # add white circular blob
        add_gaussian_spot(img, x, y, sigma, peak)
        # save and classify
        cv2.imwrite('temp.JPG', img)
        
        _, top_tokens = classify_image('temp.JPG', model)
        # find probability for token 'Yes'
        yes_prob = 0.0
        for token, probability in top_tokens:
            clean = token.strip().lower()
            if 'yes' in clean:
                yes_prob = probability
                break
        result_array[iy, ix] = [x, y, yes_prob]
        
# Save the result array to disk
array_name = f"xyp_{model}_r{sigma}_p{peak}.npy"
np.save(array_name, result_array)

# plot output
x_vals = result_array[:, :, 0].flatten()
y_vals = result_array[:, :, 1].flatten()
p_vals = result_array[:, :, 2].flatten()

plt.figure(figsize=(8, 6))
scatter = plt.scatter(x_vals, y_vals, c=p_vals, cmap='turbo')
plt.clim(0, 1)
plt.colorbar(scatter, label="yes probability")
plt.title("Probability of 'yes'")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().invert_yaxis()
plt.axis('equal')
plt.show()