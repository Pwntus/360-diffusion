import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from huggingface_hub import hf_hub_download


better_vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=better_vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

pipe.save_pretrained("./sdxl-cache", safe_serialization=True)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

# TODO - we don't need to save all of this and in fact should save just the unet, tokenizer, and config.
pipe.save_pretrained("./refiner-cache", safe_serialization=True)

safety = StableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker",
    torch_dtype=torch.float16,
)

safety.save_pretrained("./safety-cache")

hf_hub_download(
    repo_id="artificialguybr/360Redmond",
    filename="View360.safetensors",
    local_dir="lora-cache",
    cache_dir="lora-cache",
)
