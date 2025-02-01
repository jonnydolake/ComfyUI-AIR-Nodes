import numpy as np
import torch
import cv2

class LineDetection:
    """
    A node that detects straight lines in an image and outputs an image with the lines drawn as white on a black background.
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
                "threshold": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "slider"
                }),
                "min_line_length": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "slider"
                }),
                "max_line_gap": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_lines"
    CATEGORY = "AIR Nodes"

    def detect_lines(self, image, threshold, min_line_length, max_line_gap):
        """
        Detect lines in the input image and draw them as white lines on a black background.
        """
        # Ensure the input image tensor has the correct shape (batch_size, height, width, channels)
        if len(image.shape) != 4 or image.shape[3] != 3:
            raise ValueError("Input image must have shape (batch_size, height, width, 3) and be an RGB image.")

        # Convert the input image tensor to a numpy array
        image_np = image.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)

        # Process each image in the batch
        output_images = []
        for img in image_np:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Detect lines using HoughLinesP
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

            # Create a black background image with the same shape as the input image
            output_image = np.zeros_like(img)

            # Ensure the output_image is contiguous and has the correct layout
            output_image = np.ascontiguousarray(output_image, dtype=np.uint8)

            # Draw the detected lines on the black background
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(output_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

            output_images.append(output_image)

        # Convert the output images back to a tensor
        output_images = np.stack(output_images, axis=0)
        output_images = output_images.astype(np.float32) / 255.0
        output_images = torch.from_numpy(output_images)

        return (output_images,)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "LineDetection": LineDetection
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LineDetection": "Line Detection"
}