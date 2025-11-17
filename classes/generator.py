from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
import torch

class Generator:
    def __init__(self, pipe: StableDiffusionPipeline | StableDiffusionImg2ImgPipeline):
        self.pipe = pipe

    def generate(self, prompt: str, num_inference_steps: int = 50, guidance_scale: float = 7.5, strength : float = 0.7) -> torch.Tensor:
        print(f"Generating image with prompt: {prompt}")
        return None