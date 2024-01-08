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


sudo cog predict -i prompt="a photo of TOK" -i controlnet_image=@person.jpg -i image=@person.jpg 

! The pipeline cant have no normale image but a controlnet image!