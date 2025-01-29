import torch
import math
from PIL import Image
from .target_location import tensor2pil, pil2tensor

def RGB2RGBA(image: Image, mask: Image) -> Image:
    (R, G, B) = image.convert('RGB').split()
    return Image.merge('RGBA', (R, G, B, mask.convert('L')))

def reverseimagebatch(images):
    reversed_images = torch.flip(images, [0])
    return (reversed_images, )


class parallax_test:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        return {
            "required": {
                "image": ("IMAGE",),
                "frames": ("INT", {"default": 10, "min": 1, "max": 99999, "step": 1}),
                "x": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1}),
                "y": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1}),
                "zoom": ("FLOAT", {"default": 1, "min": 0.010, "max": 100, "step": 0.001}),
                "aspect_ratio": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
                "parallax_strength": ("FLOAT", {"default": 0.50, "min": 0.01, "max": 1.00, "step": 0.01}),
                "static_background": ("BOOLEAN", { "default": False }),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'layer_image_transform'
    CATEGORY = 'AIR Nodes'

    def layer_image_transform(self, image, frames, x, y, zoom, aspect_ratio, parallax_strength, static_background):

        l_images = []
        l_masks = []
        prev_comp_images = []
        comp_images = []

        mask_count = 0

        temp_x = 0
        temp_y = 0
        temp_zoom = 1.00
        prlx_x = 0
        prlx_y = 0
        prlx_zoom = 0.00

        # Convert input images to lists of layers and masks
        for l in image:
            temp_img_list = []
            temp_mask_list = []
            for frame in range(frames):
                temp_img_list.append(torch.unsqueeze(l, 0))
                m = tensor2pil(l)
                if m.mode == 'RGBA':
                    temp_mask_list.append(m.split()[-1])
            l_images.append(temp_img_list)
            l_masks.append(temp_mask_list)

        for img in l_images:
            temp_x += prlx_x
            temp_y += prlx_y
            temp_zoom += prlx_zoom

            ret_images = []
            ret_masks = []

            for i in range(len(img)):
                if mask_count == 0 and static_background:
                    temp_x = 0
                    temp_y = 0
                    temp_zoom = 1.00

                if i == 0:
                    temp_x = 0
                    temp_y = 0
                    temp_zoom = 1.00

                layer_image = img[i] if i < len(img) else img[-1]
                _image = tensor2pil(layer_image).convert('RGB')

                if i < len(l_masks[mask_count]):
                    _mask = l_masks[mask_count][i]
                else:
                    _mask = Image.new('L', size=_image.size, color='white')
                _image_canvas = Image.new('RGB', size=_image.size, color='black')
                _mask_canvas = Image.new('L', size=_mask.size, color='black')
                orig_layer_width = _image.width
                orig_layer_height = _image.height
                target_layer_width = int(orig_layer_width * temp_zoom)
                target_layer_height = int(orig_layer_height * temp_zoom * aspect_ratio)

                # Zoom
                _image = _image.resize((target_layer_width, target_layer_height))
                _mask = _mask.resize((target_layer_width, target_layer_height))

                # Composite layer
                paste_x = (orig_layer_width - _image.width) // 2 + temp_x
                paste_y = (orig_layer_height - _image.height) // 2 + temp_y
                _image_canvas.paste(_image, (paste_x, paste_y))
                _mask_canvas.paste(_mask, (paste_x, paste_y))

                if tensor2pil(layer_image).mode == 'RGBA':
                    _image_canvas = RGB2RGBA(_image_canvas, _mask_canvas)

                ret_images.append(pil2tensor(_image_canvas))
                ret_masks.append(pil2tensor(_mask_canvas))

                temp_x += x + prlx_x
                temp_y += y + prlx_y
                temp_zoom += zoom - 1 + prlx_zoom

            # Combine layers without using external blend functions
            if mask_count == 0:
                prev_comp_images = ret_images
                comp_images = ret_images
            else:
                comp_images = []
                for i in range(len(prev_comp_images)):
                    _canvas = tensor2pil(prev_comp_images[i]).convert('RGBA')
                    _layer = tensor2pil(ret_images[i])
                    _mask = tensor2pil(ret_masks[i])

                    # Perform manual blending (normal blend mode)
                    _comp = Image.new("RGBA", _canvas.size, (0, 0, 0, 0))
                    _comp.paste(_canvas, (0, 0))
                    _comp.paste(_layer, (0, 0), mask=_mask)
                    comp_images.append(pil2tensor(_comp))

                prev_comp_images = comp_images

            if mask_count == 0 and static_background:
                mask_count += 1
                temp_x = 0
                temp_y = 0
                temp_zoom = 0
                continue

            mask_count += 1

            prlx_x += math.ceil(x*parallax_strength)
            prlx_y += math.ceil(y*parallax_strength)
            prlx_zoom += (zoom-1)*parallax_strength

            temp_x = 0
            temp_y = 0
            temp_zoom = 0

        return (torch.cat(comp_images, dim=0),)


class easy_parallax:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        vertical_options = ['None', 'Pan Up', 'Pan Down']
        horizontal_options = ['None', 'Pan Left', 'Pan Right']
        zoom_options = ['None', 'Zoom In', 'Zoom Out']
        return {
            "required": {
                "image": ("IMAGE",),
                #"frames": ("INT", {"default": 10, "min": 1, "max": 99999, "step": 1}),
                "vertical_pan": (vertical_options, {"default": 'Pan Up'}),
                "horizontal_pan": (horizontal_options,),
                "camera_zoom": (zoom_options, ),
                "parallax_strength": ("FLOAT", {"default": 0.50, "min": 0.01, "max": 1.00, "step": 0.01}),
                "keep_background_static": ("BOOLEAN", {"default": False}),

            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'parallax_transform'
    CATEGORY = 'AIR Nodes'

    def parallax_transform(self, image, vertical_pan, horizontal_pan, camera_zoom, parallax_strength, keep_background_static):

        #set up
        frames = 25
        if vertical_pan == "None":
            y = 0
        elif vertical_pan == "Pan Up":
            y = 10
        elif vertical_pan == "Pan Down":
            y = -10

        if horizontal_pan == "None":
            x = 0
        elif horizontal_pan == "Pan Left":
            x = 10
        elif horizontal_pan == "Pan Right":
            x = -10

        if camera_zoom == "None":
            zoom = 1.00
        elif camera_zoom == "Zoom In":
            zoom = 1.020
        elif camera_zoom == "Zoom Out":
            x *= -1
            y *= -1
            zoom = 1.020


        l_images = []
        l_masks = []
        prev_comp_images = []
        comp_images = []

        mask_count = 0

        temp_x = 0
        temp_y = 0
        temp_zoom = 1.00
        prlx_x = 0
        prlx_y = 0
        prlx_zoom = 0.00
        aspect_ratio = 1.00

        # Convert input images to lists of layers and masks
        for l in image:
            temp_img_list = []
            temp_mask_list = []
            for frame in range(frames):
                temp_img_list.append(torch.unsqueeze(l, 0))
                m = tensor2pil(l)
                if m.mode == 'RGBA':
                    temp_mask_list.append(m.split()[-1])
            l_images.append(temp_img_list)
            l_masks.append(temp_mask_list)

        for img in l_images:
            temp_x += prlx_x
            temp_y += prlx_y
            temp_zoom += prlx_zoom

            ret_images = []
            ret_masks = []

            for i in range(len(img)):
                if mask_count == 0 and keep_background_static:
                    temp_x = 0
                    temp_y = 0
                    temp_zoom = 1.00

                if i == 0:
                    temp_x = 0
                    temp_y = 0
                    temp_zoom = 1.00

                layer_image = img[i] if i < len(img) else img[-1]
                _image = tensor2pil(layer_image).convert('RGB')

                if i < len(l_masks[mask_count]):
                    _mask = l_masks[mask_count][i]
                else:
                    _mask = Image.new('L', size=_image.size, color='white')
                _image_canvas = Image.new('RGB', size=_image.size, color='black')
                _mask_canvas = Image.new('L', size=_mask.size, color='black')
                orig_layer_width = _image.width
                orig_layer_height = _image.height
                target_layer_width = int(orig_layer_width * temp_zoom)
                target_layer_height = int(orig_layer_height * temp_zoom * aspect_ratio)

                # Zoom
                _image = _image.resize((target_layer_width, target_layer_height))
                _mask = _mask.resize((target_layer_width, target_layer_height))

                # Composite layer
                paste_x = (orig_layer_width - _image.width) // 2 + temp_x
                paste_y = (orig_layer_height - _image.height) // 2 + temp_y
                _image_canvas.paste(_image, (paste_x, paste_y))
                _mask_canvas.paste(_mask, (paste_x, paste_y))

                if tensor2pil(layer_image).mode == 'RGBA':
                    _image_canvas = RGB2RGBA(_image_canvas, _mask_canvas)

                ret_images.append(pil2tensor(_image_canvas))
                ret_masks.append(pil2tensor(_mask_canvas))

                temp_x += x + prlx_x
                temp_y += y + prlx_y
                temp_zoom += zoom - 1 + prlx_zoom

            # Combine layers without using external blend functions
            if mask_count == 0:
                prev_comp_images = ret_images
                comp_images = ret_images
            else:
                comp_images = []
                for i in range(len(prev_comp_images)):
                    _canvas = tensor2pil(prev_comp_images[i]).convert('RGBA')
                    _layer = tensor2pil(ret_images[i])
                    _mask = tensor2pil(ret_masks[i])

                    # Perform manual blending (normal blend mode)
                    _comp = Image.new("RGBA", _canvas.size, (0, 0, 0, 0))
                    _comp.paste(_canvas, (0, 0))
                    _comp.paste(_layer, (0, 0), mask=_mask)
                    comp_images.append(pil2tensor(_comp))

                prev_comp_images = comp_images

            if mask_count == 0 and keep_background_static:
                mask_count += 1
                temp_x = 0
                temp_y = 0
                temp_zoom = 0
                continue

            mask_count += 1

            prlx_x += math.ceil(x * parallax_strength)
            prlx_y += math.ceil(y * parallax_strength)
            prlx_zoom += (zoom - 1) * parallax_strength

            temp_x = 0
            temp_y = 0
            temp_zoom = 0

        if camera_zoom == "Zoom Out":
            return (torch.cat(comp_images, dim=0),)
        else:
            return (torch.cat(comp_images[::-1], dim=0),)



NODE_CLASS_MAPPINGS = {
    "ParallaxTest": parallax_test,
    "easy_parallax": easy_parallax,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParallaxTest": "Parallax Test",
    "easy_parallax": "Easy Parallax",
}