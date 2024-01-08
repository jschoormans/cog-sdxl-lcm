from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, LCMScheduler
import torch
from diffusers.utils import load_image

# a test script to try ip adapter and sd1.5 
# this should also have densepose and loras!

# 0 add ip_adapter [x]
# 1 add inpaint [x]
# 2 load other LORAs [ ]
# 3 add controlnet Canny to test [x]
# 4 add densePose controlnet [x]
# 5 add LCM [x] 

# pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float
#                                                   ).to("cuda")

# pipeline = StableDiffusionInpaintPipeline.from_pretrained(
#     "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
# )

if False:
    controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
    controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
else:
    print('using dense pose...')
    controlnet = ControlNetModel.from_single_file('densepose/densepose.safetensors',use_safetensors=True)
    
pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float)
pipeline.to("cuda")



image = load_image("https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/53b91522-790b-47f7-b078-93bb21e94a4f/width=450/1.jpeg")
mask = load_image("densemask.jpeg")

control_image = load_image("https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/33d861e5-996f-4796-b38e-9e6cc5f30803/width=450/1d.jpeg")
ip_image = load_image("https://www.sophiatolli.com/uploads/images/products/513/0a737fd6_3b2f_409e_8c09_8a30f948831f.jpg?w=670")

image = image.resize((512, 768))
mask = mask.resize((512, 768))
control_image = control_image.resize((512,768))


pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")


### LCM part
pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()



generator = torch.Generator(device="cpu").manual_seed(13)
images = pipeline(
    prompt='woman wearing a white shirt dress, best quality, high quality', 
    image = image,
    mask_image = mask,
    control_image = control_image,
    ip_adapter_image=ip_image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality, nsfw, nude, naked", 
    num_inference_steps=12,
    generator=generator,
    strength=1,
    guidance_scale=1,
).images
images[0]
images[0].save('experiment-result.jpg')