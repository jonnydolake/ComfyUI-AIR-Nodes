import cv2
import numpy as np
import torch
from PIL import Image

class MangaPanelSegmentationNode:
    """
    A ComfyUI node to create a mask of panels in a manga page.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input types for the node.
        """
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "AIR Nodes"

    def process_image(self, image):
        """
        Process the image to create a mask of the panels.
        """

        # Convert the input image tensor to a numpy array
        image_np = image.cpu().numpy()[0]  # Assuming batch size is 1
        image_np = (image_np * 255).astype(np.uint8)  # Convert to 8-bit image

        # Convert to grayscale
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Apply thresholding to create a binary image
        ret, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Create a mask for the contours
        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        if len(contours) > 0:  # Check if any contours were found
            cv2.drawContours(contour_mask, contours, -1, 255, -1)
        else:
            print("Warning: No contours found in the image. Returning an empty mask.")

        # Convert the mask back to a tensor
        contour_mask_tensor = torch.from_numpy(contour_mask).float() / 255.0  # Normalize to [0, 1]
        contour_mask_tensor = contour_mask_tensor.unsqueeze(0).unsqueeze(-1)  # Add batch and channel dimensions

        # Ensure the tensor has 3 channels for compatibility with ComfyUI's image preview
        contour_mask_tensor = contour_mask_tensor.repeat(1, 1, 1, 3)  # Repeat the single channel to create 3 channels


        return (contour_mask_tensor,)


# Register the node
NODE_CLASS_MAPPINGS = {
    "MangaPanelSegmentationNode": MangaPanelSegmentationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MangaPanelSegmentationNode": "Manga Panel Segmentation Node",
}
