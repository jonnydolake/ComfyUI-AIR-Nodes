import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from .target_location import pil2mask

def fill_region(image):

    from scipy.ndimage import binary_fill_holes
    image = image.convert("L")
    binary_mask = np.array(image) > 0
    filled_mask = binary_fill_holes(binary_mask)
    filled_image = Image.fromarray(filled_mask.astype(np.uint8) * 255, mode="L")
    return ImageOps.invert(filled_image.convert("RGB"))

class Mask_Fill_Region:

    def __init__(self):
        pass
        #self.WT = WAS_Tools_Class()

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks": ("MASK",),
                    }
                }

    CATEGORY = "AIR Nodes"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "fill_region"

    def fill_region(self, masks):
        if masks.ndim > 3:
            regions = []
            for mask in masks:
                mask_np = np.clip(255. * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(mask_np, mode="L")
                region_mask = fill_region(pil_image)
                region_tensor = pil2mask(region_mask).unsqueeze(0).unsqueeze(1)
                regions.append(region_tensor)
            regions_tensor = torch.cat(regions, dim=0)
            return (regions_tensor,)
        else:
            mask_np = np.clip(255. * masks.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(mask_np, mode="L")
            region_mask = fill_region(pil_image)
            region_tensor = pil2mask(region_mask).unsqueeze(0).unsqueeze(1)
            return (region_tensor,)

# Register the node
NODE_CLASS_MAPPINGS = {
    "Mask_Fill_Region": Mask_Fill_Region,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mask_Fill_Region": "Mask Fill Region",
}
