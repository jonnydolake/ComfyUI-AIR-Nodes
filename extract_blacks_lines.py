from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np
import torch
import scipy.ndimage
from .target_location import pil2tensor, tensor2pil, get_image_size


class extract_lines:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image" :("IMAGE", ),
                "line_threshold": ("INT",{
                    "default": 193, "min": 1, "max": 255, "step":1, "display": "number"
                }),
                "tolerance": ("INT", {
                    "default": 50, "min": 0, "max": 100, "step": 1, "display": "number"
                }),
                "grow_mask": ("INT", {
                    "default": 0, "min": 0, "max": 10, "step": 1, "display": "number"
                }),

            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)

    FUNCTION = "run"

    CATEGORY = "AIR Nodes"

    def run(self, image, line_threshold, tolerance, grow_mask):
        size = get_image_size(image)
        width_B = int(size[0])
        height_B = int(size[1])
        empty_white = Image.new('RGB', (width_B, height_B), color='white')


        # Assuming image shape is (batch_size, height, width, channels)
        image_np = image.cpu().numpy()

        # If batch_size is 1, we can squeeze it out
        if image_np.shape[0] == 1:
            image_np = image_np.squeeze(0)

        # Convert to uint8 and scale to 0-255
        rgb_img = (image_np * 255).astype(np.uint8)

        # Convert RGB to HSV
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

        # Define HSV
        lower = np.array([0, 0, 0])
        upper = np.array([179, tolerance, 130])  # Adjust the upper value for black

        # Create masks for black
        color_mask = cv2.inRange(hsv_img, lower, upper)
        pil_color_mask = Image.fromarray(color_mask)

        # Convert the mask to a tensor
        color_mask_tensor = torch.tensor(color_mask, dtype=torch.float32)
        final_color_mask = torch.from_numpy(np.array(color_mask_tensor).astype(np.float32) / 255.0).unsqueeze(0)

        #composited_image = composite_masked(empty_white, image, resize_source=False, mask=final_color_mask)
        composited_image = Image.composite(empty_white, tensor2pil(image), ImageOps.invert(pil_color_mask))

        composited_image = np.array(composited_image)
        #up until here everything is GOOD!


        bw_composited_image = cv2.cvtColor(composited_image, cv2.COLOR_RGB2GRAY)

        _, binary_image = cv2.threshold(bw_composited_image, line_threshold, 255, cv2.THRESH_BINARY)

        #binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)

        #binary_image = HWC3(binary_image)
        #bw_binary_image = cv2.cvtColor(binary_image, cv2.COLOR_RGB2GRAY)


        kernel = np.ones((5, 5), np.uint8)

        binary_grow = cv2.erode(binary_image, kernel, iterations=grow_mask)

        final_composited_image = Image.composite(empty_white, tensor2pil(image), Image.fromarray(binary_grow, mode='L'))

        #final_output = ImageEnhance.Color(final_composited_image).enhance(0)

        return (pil2tensor(final_composited_image), final_color_mask,)


NODE_CLASS_MAPPINGS = {"ExtractBlackLines": extract_lines,}

NODE_DISPLAY_NAME_MAPPINGS = {"ExtractBlackLines": "Extract Black Lines",}
