import folder_paths
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch
import os

folder_paths.folder_names_and_paths["yolov8"] = ([os.path.join(folder_paths.models_dir, "yolov8")], folder_paths.supported_pt_extensions)

class Yolov8Detection:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",), 
                "model_name": (folder_paths.get_filename_list("yolov8"), ),
                "mask_output_type": (["segmented", "bbox"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("image", "masks", "if_empty_boolean")
    FUNCTION = "detect"
    CATEGORY = "AIR Nodes"

    def detect(self, image, model_name, mask_output_type):

        empty_mask = False

        # Convert tensor to numpy array and then to PIL Image
        image_tensor = image
        image_np = image_tensor.cpu().numpy()  # Change from CxHxW to HxWxC for Pillow
        image = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))  # Convert float [0,1] tensor to uint8 image
        
        print(f'model_path: {os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}')
        model = YOLO(f'{os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}')  # load a custom model
        results = model(image)

        if mask_output_type == "segmented":
            # Extract masks
            masks = results[0].masks
            mask_tensors = []
            if masks is not None:
                for mask in masks:
                    mask_np = mask.data.cpu().numpy()  # Convert mask to numpy array
                    mask_tensor = torch.tensor(mask_np, dtype=torch.float32)  # Convert to tensor
                    mask_tensors.append(mask_tensor)
        elif mask_output_type == "bbox":
            # Get image dimensions
            image_width, image_height = image.size

            # Initialize an empty list to store masks
            mask_tensors = []

            # Extract bounding boxes and create masks
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes in xyxy format
                for box in boxes:
                    x1, y1, x2, y2 = box
                    # Create a binary mask for the bounding box
                    mask = np.zeros((image_height, image_width), dtype=np.float32)
                    mask[int(y1):int(y2), int(x1):int(x2)] = 1.0  # Set the region inside the bounding box to 1
                    mask_tensor = torch.tensor(mask, dtype=torch.float32)  # Convert to tensor
                    mask_tensors.append(mask_tensor)

        
        # If no masks are detected, create a single black mask
        if not mask_tensors:
            black_mask = np.zeros((image_height, image_width), dtype=np.float32)
            mask_tensors.append(torch.tensor(black_mask, dtype=torch.float32))
            empty_mask = True
        



        # Stack masks into a single tensor
        mask_batch = torch.stack(mask_tensors, dim=0)  # Shape: (N, H, W)

        # Plot the image with bounding boxes
        im_array = results[0].plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

        image_tensor_out = torch.tensor(np.array(im).astype(np.float32) / 255.0)  # Convert back to CxHxW
        image_tensor_out = torch.unsqueeze(image_tensor_out, 0)

        '''
        # Prepare JSON output
        json_output = {
            "classify": [r.boxes.cls.tolist()[0] for r in results],
            "boxes": [r.boxes.xyxy.tolist() for r in results],
        }
        '''

        return (image_tensor_out, mask_batch, empty_mask)
    

NODE_CLASS_MAPPINGS = {
    "Yolov8Detection": Yolov8Detection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yolov8Detection": "Yolov8 Detection",
}
