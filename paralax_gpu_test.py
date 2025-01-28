import torch
import numpy as np
from PIL import Image
from .target_location import pil2tensor, tensor2pil
from .blendmodes import BLEND_MODES

# Optimized RGB to RGBA conversion using PyTorch
def RGB2RGBA(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(0)  # Ensure mask has a single channel
    return torch.cat((image, mask), dim=0)  # Combine RGB and mask into RGBA

# Optimized blending function
def chop_image_v2(background_image: torch.Tensor, layer_image: torch.Tensor, blend_mode: str, opacity: float) -> torch.Tensor:
    blended = BLEND_MODES[blend_mode](background_image, layer_image, opacity / 100)
    return blended.to('cpu')  # Ensure output is on CPU for further processing

class ParallaxGPUTest:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "frames": ("INT", {"default": 10, "min": 1, "max": 99999, "step": 1}),
                "x": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1}),
                "y": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1}),
                "zoom": ("FLOAT", {"default": 1, "min": 0.010, "max": 100, "step": 0.001}),
                "aspect_ratio": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "layer_image_transform"
    CATEGORY = "AIR Nodes"

    def layer_image_transform(self, image, frames, x, y, zoom, aspect_ratio):
        # Move tensors to GPU
        l_images = [l.to('cuda') for l in image]
        l_masks = []

        for l in l_images:
            masks = []
            for frame in range(frames):
                m = tensor2pil(l).convert('RGBA').split()[-1]  # Extract alpha channel
                masks.append(pil2tensor(m).to('cuda'))
            l_masks.append(masks)

        comp_images = []
        for i, layer_image in enumerate(l_images):
            temp_x, temp_y, temp_zoom = 0, 0, 1.0

            for frame in range(frames):
                current_layer = layer_image if frame < len(layer_image) else layer_image[-1]

                mask = l_masks[i][frame] if frame < len(l_masks[i]) else torch.ones_like(layer_image[0:1])

                # Transform layer
                _, h, w = current_layer.shape
                new_h, new_w = int(h * temp_zoom), int(w * temp_zoom * aspect_ratio)
                transformed_layer = torch.nn.functional.interpolate(
                    current_layer.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
                ).squeeze(0)

                transformed_mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
                ).squeeze(0)

                # Composite layer on canvas
                canvas = torch.zeros_like(layer_image)
                canvas[:, :new_h, :new_w] = transformed_layer
                mask_canvas = torch.zeros_like(mask)
                mask_canvas[:new_h, :new_w] = transformed_mask

                # Blend images
                if i > 0:
                    prev_image = comp_images[-1]
                    blended = chop_image_v2(prev_image, canvas, "normal", 100)
                    comp_images.append(blended)
                else:
                    comp_images.append(canvas)

                # Update for next frame
                temp_x += x
                temp_y += y
                temp_zoom *= zoom

        # Combine all frames into a single tensor
        final_output = torch.stack(comp_images, dim=0).to('cpu')
        return (final_output,)


NODE_CLASS_MAPPINGS = {
    "ParallaxGPUTest": ParallaxGPUTest,
    "easy_parallax": easy_parallax,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParallaxGPUTest": "Parallax GPU Test",
}