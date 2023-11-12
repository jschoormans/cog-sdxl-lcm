# Run this before you deploy it on replicate, because if you don't
# whenever you run the model, it will download the weights from the
# internet, which will take a long time.

import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

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
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", cache_dir="lcm-cache")
pipe.fuse_lora()
pipe.save_pretrained("./sdxl-cache", safe_serialization=True)


# safety = StableDiffusionSafetyChecker.from_pretrained(
#     "CompVis/stable-diffusion-safety-checker",
#     torch_dtype=torch.float16,
# )

# safety.save_pretrained("./safety-cache")
