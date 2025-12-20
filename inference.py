import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe.load_lora_weights("SherryXTChen/InstructCLIP-InstructPix2Pix")
pipe.load_lora_weights("/home/data10T/lpy/mml-proj/ckpts/ip2p_finetuned_test/pytorch_lora_weights.safetensors", weight_name="pytorch_lora_weights.safetensors")
# print("Loaded LoRA weights.")
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

url = "https://raw.githubusercontent.com/SherryXTChen/Instruct-CLIP/refs/heads/main/assets/2_input.jpg"
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image
image = download_image(url)

prompt = "remove snow"
images = pipe(prompt, image=image, num_inference_steps=20).images
images[0].save("output.jpg")