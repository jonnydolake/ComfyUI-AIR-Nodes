import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from .target_location import pil2mask

def fill_region(image, fill_only=True):

    # Convert image to grayscale
    image = image.convert("L")
    
    # Convert image to binary mask
    binary_mask = np.array(image) > 0
    
    # Initialize a mask to store the combined results
    combined_filled_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    
    # Define padding directions: top, bottom, left, right
    padding_directions = [
        ((1, 0), (0, 0)),  # Top
        ((0, 1), (0, 0)),  # Bottom
        ((0, 0), (1, 0)),  # Left
        ((0, 0), (0, 1)),  # Right
    ]
    
    # Process each edge individually
    for pad_top_bottom, pad_left_right in padding_directions:
        # Pad the mask for the current edge
        padded_mask = np.pad(binary_mask, (pad_top_bottom, pad_left_right), mode='constant', constant_values=0)
        
        # Convert the padded mask to uint8 (required by cv2.floodFill)
        padded_mask_uint8 = padded_mask.astype(np.uint8) * 255
        
        # Perform flood-fill starting from the padded edge
        h_pad, w_pad = padded_mask_uint8.shape
        if pad_top_bottom == (1, 0):  # Top edge
            for x in range(w_pad):
                if padded_mask_uint8[0, x] == 0:
                    cv2.floodFill(padded_mask_uint8, None, (x, 0), 255)
        elif pad_top_bottom == (0, 1):  # Bottom edge
            for x in range(w_pad):
                if padded_mask_uint8[h_pad-1, x] == 0:
                    cv2.floodFill(padded_mask_uint8, None, (x, h_pad-1), 255)
        elif pad_left_right == (1, 0):  # Left edge
            for y in range(h_pad):
                if padded_mask_uint8[y, 0] == 0:
                    cv2.floodFill(padded_mask_uint8, None, (0, y), 255)
        elif pad_left_right == (0, 1):  # Right edge
            for y in range(h_pad):
                if padded_mask_uint8[y, w_pad-1] == 0:
                    cv2.floodFill(padded_mask_uint8, None, (w_pad-1, y), 255)
        
        # Remove the padding
        filled_mask = padded_mask_uint8[pad_top_bottom[0]:h_pad-pad_top_bottom[1], pad_left_right[0]:w_pad-pad_left_right[1]]
        
        # Combine the results
        combined_filled_mask = np.logical_or(combined_filled_mask, filled_mask == 0)
        if not fill_only:
            combined_filled_mask = np.logical_or(combined_filled_mask, binary_mask)
    
    # Convert the combined filled mask back to an image
    filled_image = Image.fromarray(combined_filled_mask.astype(np.uint8) * 255, mode="L")

    return ImageOps.invert(filled_image.convert("RGB"))

class Mask_Fill_Region:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "return_fill_only": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "AIR Nodes"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "fill_mask"

    def fill_mask(self, masks, return_fill_only):
        if masks.ndim > 3:
            regions = []
            for mask in masks:
                mask_np = np.clip(255. * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(mask_np, mode="L")
                region_mask = fill_region(pil_image, return_fill_only)

                region_tensor = pil2mask(region_mask).unsqueeze(0).unsqueeze(1)
                regions.append(region_tensor)
            regions_tensor = torch.cat(regions, dim=0)
            return (regions_tensor,)
        else:
            mask_np = np.clip(255. * masks.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(mask_np, mode="L")
            region_mask = fill_region(pil_image, return_fill_only)

            region_tensor = pil2mask(region_mask).unsqueeze(0).unsqueeze(1)
            return (region_tensor,)


NODE_CLASS_MAPPINGS = {
    "Mask_Fill_Region": Mask_Fill_Region,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mask_Fill_Region": "Mask Fill Region",
}

