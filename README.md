# 360 Diffusion Cog model

The model is using [Openjourney v4](https://huggingface.co/prompthero/openjourney-v4) with a [360 Diffusion LoRA](https://huggingface.co/ProGamerGov/360-Diffusion-LoRA-sd-v1-5). Append the prompt with `qxj <lora:360Diffusion_v1:1>` to activate the equirectangular effect.

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="360 degree equirectangular panorama photo, interior of wooden house, qxj <lora:360Diffusion_v1:1>"
