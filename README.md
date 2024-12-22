# tiny ghosts :ghost:
The American West is known for its deserts and canyons, but is also rich in biodiversity. 
Its inhabitants include fireflies. :bug::sparkles:

Western firefly populations are small and sparse, and little is known about them. 
They often reside in remote areas (springs, arroyos, etc.) where access is limited, making their study and monitoring cumbersome.

In 2023, Cheryl Mollohan and Ron Day pioneered the use of low-light trail cameras to monitor firefly populations in Arizona over several weeks. 
Trail cameras capture flashes surprisingly well, but they produce thousands of pictures which then need to be classified (flash/no flash). 

## Purpose
`tiny-ghosts` is a project for automated detection of firefly flashes in trail cameras images using computer vision.
In this context, finding which pictures show firefly flashes is not an easy task.
The pictures have low contrast, show a complex background, with uneven and changing illumination (moon, lightning), and various kinds of hotspots and artifacts (headlights).

![typical trail camera picture showing two firefly flashes in the center](jpg/flash/DSCF0304.JPG)

## Approaches
I am trying two main approaches:
1. Traditional image binarization followed by neural network classification;
2. Use of multimodal LLMs.

### Vision Models
The past two years have seen the emergence of pretrained multimodal LLMs, which excel at a variety of tasks.
It turns out they are also pretty good at identifying fireflies.

#### GPT-4o
The most successful approach has been using the OpenAI API to pass images to their vision models (typically GPT-4o) and return a yes or no for the presence of flashes.
I have found that the model becomes increasingly accurate when images are split into patches (512x512 to match the API constraints), although the processing is slower.
A downside of this approach is the non-negligible cost: over $5 for 1000 pictures (passed as 8 patches).

I'm investigating using the logits/softmax value to adapt the threshold of positive/negative, possibly away from p = 0.5.

#### Llava
[Llava](https://llava-vl.github.io) is a vision model based on Llama. 
It performs reasonably well to identify patches, although not as well as 4o. 
It is free and runs on a MacBook Pro M2.

### Blob detection + CNN classification
Based on [`firefleye-net`](https://github.com/rapsar/firefl-eye-net). 
Coming soon.

## Installation
1. Clone this repo: `git clone https://github.com/your-username/tiny-ghosts.git`
2. Set up the virtual environment: `tgenv`
3. Run the main script to process images.
