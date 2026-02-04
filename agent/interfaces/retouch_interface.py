import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import (
    DDPMScheduler, 
    AutoencoderKL
)
from transformers import CLIPTextModel

from models.controlnet import ControlNetModel
from models.pipeline_controlnet import StableDiffusionControlNetPipeline
from models.unet_2d_condition import UNet2DConditionModel
from models.utils import UnetConvout

class RetouchInterface:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32

        self.text_encoder = CLIPTextModel.from_pretrained(
            "sd2-community/stable-diffusion-2-1",
            subfolder="text_encoder",
            cache_dir="../model",
        ).to(self.device, dtype=self.torch_dtype)

        self.vae = AutoencoderKL.from_pretrained(
            "sd2-community/stable-diffusion-2-1", 
            subfolder="vae", 
            cache_dir="../model",
        ).to(self.device, dtype=self.torch_dtype)

        self.controlnet = ControlNetModel.from_pretrained(
            "../model/ckpt/controlnet",
            conditioning_channels=4
        ).to(self.device, dtype=self.torch_dtype)

        self.unet = UNet2DConditionModel.from_pretrained(
            "sd2-community/stable-diffusion-2-1", 
            subfolder="unet", 
            cache_dir="../model",
        ).to(self.device, dtype=self.torch_dtype)

        self.unet_conv_out = UnetConvout().to(self.device, dtype=self.torch_dtype)
        self.unet_conv_out.load_state_dict(torch.load(
            "../model/ckpt/unet_conv_out.pth",
            weights_only=True,
        ))

        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "sd2-community/stable-diffusion-2-1",
            vae=self.vae,
            text_encoder=self.text_encoder,
            controlnet=self.controlnet,
            unet=self.unet,
            safety_checker=None,
            cache_dir="../model",
            torch_dtype=self.torch_dtype,
        ).to(self.device)

    def process(self, original_img, param_map):
        """
        original_img: RGB numpy array (0-255)
        param_map: 512x512x4 numpy array (-1 to 1)
        """
        # Numpy -> PIL
        image = Image.fromarray(original_img.astype('uint8')).convert("RGB")
        original_width, original_height = image.size

        img_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])

        # Numpy -> Tensor (512, 512, 4) -> (1, 4, 512, 512)
        param_map_tensor = torch.from_numpy(param_map).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.torch_dtype)

        with torch.no_grad():
            # VAE 编码原图
            image_tensor = img_transform(image).unsqueeze(0).to(self.device, dtype=self.torch_dtype)
            latents = self.pipeline.vae.encode(image_tensor).latent_dist.sample()
            latents *= self.pipeline.vae.config.scaling_factor

            generator = torch.Generator(device=self.device).manual_seed(42)
            noisy_latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.torch_dtype, generator=generator)

            output = self.pipeline(
                prompt="",
                image=param_map_tensor, 
                latents=noisy_latents,
                original_img_vae=latents,
                num_inference_steps=50,
                generator=generator,
                guidance_scale=0,
                unet_conv_out=self.unet_conv_out,
            )

        res_image = output[0].resize((original_width, original_height), resample=Image.LANCZOS)

        return np.array(res_image)