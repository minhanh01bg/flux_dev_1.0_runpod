'''
Contains the handler function that will be called by the serverless.
'''

import io
from PIL import Image, ImageOps
import base64

import torch
from diffusers import FluxPipeline
import runpod

from huggingface_hub import login
login(token="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")
torch.cuda.empty_cache()


def pil_to_base64(result):
    buffered_image = io.BytesIO()
    if "exif" in result.info:
        exif = result.info.get("exif")
        result = ImageOps.exif_transpose(result)
    result.save(buffered_image, format="WebP", lossless=True)
    image_base64 = base64.b64encode(buffered_image.getvalue()).decode('utf-8')
    return image_base64


# ------------------------------- Model Handler ------------------------------ #


class ModelHandler:
    def __init__(self):
        # self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model_id = "black-forest-labs/FLUX.1-dev"

    def load_models(self):
        self.pipeline = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16
        ).to(self.device)

    def pipe(self,
             prompt: str,
             negative_prompt: str,
             num_inference_steps: int,
             height: int,
             width: int,
             guidance_scale: float,
             num_images_per_prompt: int,):
        print(f"prompt: {prompt}")
        print(f"negative_prompt: {negative_prompt}")
        print(f"num_inference_steps: {num_inference_steps}")
        print(f"height: {height}")
        print(f"width: {width}")
        print(f"guidance_scale: {guidance_scale}")
        print(f"num_images_per_prompt: {num_images_per_prompt}")

        generator = torch.manual_seed(42)

        image = self.pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            generator=generator,
            max_sequence_length=150,
            num_images_per_prompt=num_images_per_prompt
        ).images
        return image

MODELS = ModelHandler()
MODELS.load_models()

# ---------------------------------- Helper ---------------------------------- #
import tempfile

def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]
    print(f"1. job_input: {job_input}")

    # Input validation
    # validated_input = validate(job_input, INPUT_SCHEMA)

    # if 'errors' in validated_input:
    #     return {"error": validated_input['errors']}
    # data = validated_input['validated_input']

    print(f"2. job_input: {job_input}")
    label = job_input.get('label', "student")
    num_images_per_prompt = 1
    if label=="teacher":
        num_images_per_prompt = 4

    
    images = MODELS.pipe(
        prompt=job_input.get('prompt'),
        negative_prompt=job_input.get('negative_prompt',""),
        num_inference_steps=job_input.get('num_inference_steps', 50),
        height=job_input.get('height', 1024),
        width=job_input.get('width', 1024),
        guidance_scale=job_input.get('guidance_scale', 3.5),
        num_images_per_prompt=num_images_per_prompt
    )
    print(f"process ok")
    if label=="teacher":
        base64_images = [pil_to_base64(img) for img in images[0:]]

        print(f"return ouput")
        results = {
            "image_urls": base64_images,
        }
    else:
        image = images[0]
        image_base64 = pil_to_base64(image)
        results = {
            "image_url": image_base64,
        }
    return results

runpod.serverless.start({"handler": generate_image})