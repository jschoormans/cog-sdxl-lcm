# Cog-SDXL-LCM

This is an implementation of Stability AI's [SDXL](https://github.com/Stability-AI/generative-models) as a [Cog](https://github.com/replicate/cog) model with LCM LoRA for faster inference.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

Download weights first to apply the LCM-LoRA for SDXL
```bash
cog run python script/download_weights.py
```

Then for predictions,

```bash
cog predict -i prompt="a photo of TOK"
```

```bash
cog train -i input_images=@example_datasets/__data.zip -i use_face_detection_instead=True
```

```bash
cog run -p 5000 python -m cog.server.http
```


sudo cog predict -i prompt="a photo of TOK" -i controlnet_image=@image.jpg -i image=@image.jpg -i mask=@mask.jpg

sudo cog predict -i prompt="a man wearing a TOK sweater" -i controlnet_image=@image.jpg -i image=@image.jpg -i mask=@mask.jpg -i prompt_strength=1.0 -i replicate_weights=https://replicate.delivery/pbxt/97WFj7UpFVofFSAmn3Ztt3CEM4rWG1lfds7kSKofv2N820UkA/trained_model.tar


! The pipeline cant have no normale image but a controlnet image!

sudo cog push r8.im/jschoormans/sdxl-lcm-openpose