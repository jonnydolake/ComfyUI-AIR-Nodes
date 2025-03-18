import torch
import numpy as np
from PIL import Image
from .target_location import tensor2pil, pil2tensor


def multiply_blend(image1, image2):
    """
    Applies multiply blending mode to two images.

    Args:
        image1: PIL Image object (background).
        image2: PIL Image object (foreground/blend layer).

    Returns:
        A new PIL Image object with the result of the blending.
    """

    img1_np = np.array(image1).astype(float) / 255.0
    img2_np = np.array(image2).astype(float) / 255.0

    result_np = img1_np * img2_np

    result_img = Image.fromarray(np.uint8(result_np * 255))
    return result_img

class displace_image:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        return {
            "required": {
                "image": ("IMAGE",),
                "displace_by": ("INT", {"default": 1, "min": 0, "max": 512, "step": 1}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'layer_image_transform'
    CATEGORY = 'AIR Nodes'

    def layer_image_transform(self, image, displace_by):

        pos_displace = displace_by
        neg_displace = -displace_by
        ret_images = []

        for img in image:
            _image = tensor2pil(img)

            image_canvas1 = Image.new('RGB', size=_image.size, color='white')
            image_canvas2 = Image.new('RGB', size=_image.size, color='white')
            image_canvas3 = Image.new('RGB', size=_image.size, color='white')
            image_canvas4 = Image.new('RGB', size=_image.size, color='white')

            # Composite
            image_canvas1.paste(_image, (pos_displace, 0))
            image_canvas2.paste(_image, (neg_displace, 0))
            image_canvas3.paste(_image, (0, pos_displace))
            image_canvas4.paste(_image, (0, neg_displace))

            #Multiply
            blend1= multiply_blend(image_canvas1,image_canvas2)
            blend2 = multiply_blend(blend1, image_canvas3)
            blend3 = multiply_blend(blend2, image_canvas4)

            ret_images.append(pil2tensor(blend3))


        return (torch.cat(ret_images, dim=0),)



NODE_CLASS_MAPPINGS = {
    "DisplaceImage": displace_image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DisplaceImage": "Displace Image",
}