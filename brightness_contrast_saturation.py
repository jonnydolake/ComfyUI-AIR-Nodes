import torch

class BrightnessContrastSaturation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct'
    CATEGORY = 'AIR Nodes'

    def color_correct(self, image, brightness, contrast, saturation):
        # Make a copy of the input tensor to avoid modifying the original
        #ret_images = image.clone()
        ret_images = image

        # Apply brightness (scale pixel values)
        if brightness != 1:
            ret_images = ret_images * brightness

        # Apply contrast
        if contrast != 1:
            # Calculate mean luminance per image in the batch
            mean = torch.mean(ret_images, dim=(1, 2), keepdim=True)
            ret_images = (ret_images - mean) * contrast + mean

        # Apply saturation
        if saturation != 1:
            # Convert to grayscale by taking mean across channels
            grayscale = ret_images.mean(dim=3, keepdim=True)
            # Interpolate between grayscale and color
            ret_images = torch.lerp(grayscale, ret_images, saturation)

        # Clamp values to valid range [0, 1]
        ret_images = torch.clamp(ret_images, 0, 1)

        return (ret_images,)


NODE_CLASS_MAPPINGS = {
    "BrightnessContrastSaturation": BrightnessContrastSaturation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BrightnessContrastSaturation": "Brightness Contrast Saturation",
}