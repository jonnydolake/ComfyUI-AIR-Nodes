import torch
import copy
import numpy as np
from PIL import Image
from .target_location import tensor2pil, pil2tensor, composite, composite_masked, pil2mask
from .blendmodes import *


#from LayerStyles
def RGB2RGBA(image:Image, mask:Image) -> Image:
    (R, G, B) = image.convert('RGB').split()
    return Image.merge('RGBA', (R, G, B, mask.convert('L')))

def chop_image_v2(background_image:Image, layer_image:Image, blend_mode:str, opacity:int) -> Image:

    backdrop_prepped = np.asfarray(background_image.convert('RGBA'))
    source_prepped = np.asfarray(layer_image.convert('RGBA'))
    blended_np = BLEND_MODES[blend_mode](backdrop_prepped, source_prepped, opacity / 100)

    # final_tensor = (torch.from_numpy(blended_np / 255)).unsqueeze(0)
    # return tensor2pil(_tensor)

    return Image.fromarray(np.uint8(blended_np)).convert('RGB')


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

            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'layer_image_transform'
    CATEGORY = 'AIR Nodes'

    def layer_image_transform(self, image, frames, x, y, zoom, aspect_ratio,):

        l_images = []
        l_masks = []
        #ret_images = []
        #ret_masks = []
        prev_comp_images = []
        comp_images = []

        mask_count = 0

        temp_x = 0
        temp_y = 0
        temp_zoom = 1.00
        prlx_x = 0
        prlx_y = 0
        prlx_zoom = 0.00

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

                # zoom
                _image = _image.resize((target_layer_width, target_layer_height))
                _mask = _mask.resize((target_layer_width, target_layer_height))

                # composite layer
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

                #ret_images.append(pil2tensor(_image_canvas))


            #print("X= " + str(temp_x) + "//  Y= " + str(temp_y))
            if mask_count == 0:
                print("TEST MASK!!!!!")
                prev_comp_images = ret_images
                comp_images = ret_images
            else:
                comp_images = []
                for i in range(len(prev_comp_images)):
                    _canvas = tensor2pil(prev_comp_images[i]).convert('RGBA')
                    _layer = tensor2pil(ret_images[i])
                    _mask = tensor2pil(ret_masks[i])

                    _comp = copy.copy(_canvas)
                    print(_comp.size)
                    #_comp = prev_comp_images[i]
                    #print(_comp)
                    _compmask = Image.new("RGBA", _comp.size, color='black')
                    _comp.paste(_layer, (0, 0))
                    _compmask.paste(_mask, (0, 0))
                    _compmask = _compmask.convert('L')
                    _comp = chop_image_v2(_canvas, _comp, "normal", 100)
                    #a_img = composite_masked(prev_comp_images[i], ret_images[i], mask=ret_masks[i])
                    print("TEST!!!!!")

                    # composition background
                    _canvas.paste(_comp, mask=_compmask)
                    #print(a_img)
                    comp_images.append(pil2tensor(_canvas))
                prev_comp_images = comp_images

            mask_count += 1

            prlx_x += x
            prlx_y += y
            prlx_zoom += zoom - 1

            temp_x = 0
            temp_y = 0
            temp_zoom = 0

        print(comp_images)
        return (torch.cat(comp_images, dim=0),)
        #return (comp_images,)


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
                "frames": ("INT", {"default": 10, "min": 1, "max": 99999, "step": 1}),
                "vertical_pan": (vertical_options,),
                "horizontal_pan": (horizontal_options,),
                "zoom": (zoom_options,),

            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'parallax_transform'
    CATEGORY = 'AIR Nodes'

    def parallax_transform(self, image, frames, vertical_pan, horizontal_pan, zoom,):

        l_images = []
        l_masks = []
        #ret_images = []
        #ret_masks = []
        prev_comp_images = []
        comp_images = []

        mask_count = 0

        temp_x = 0
        temp_y = 0
        temp_zoom = 1.00
        prlx_x = 0
        prlx_y = 0
        prlx_zoom = 0.00

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

                # zoom
                _image = _image.resize((target_layer_width, target_layer_height))
                _mask = _mask.resize((target_layer_width, target_layer_height))

                # composite layer
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

                #ret_images.append(pil2tensor(_image_canvas))


            #print("X= " + str(temp_x) + "//  Y= " + str(temp_y))
            if mask_count == 0:
                print("TEST MASK!!!!!")
                prev_comp_images = ret_images
                comp_images = ret_images
            else:
                comp_images = []
                for i in range(len(prev_comp_images)):
                    _canvas = tensor2pil(prev_comp_images[i]).convert('RGBA')
                    _layer = tensor2pil(ret_images[i])
                    _mask = tensor2pil(ret_masks[i])

                    _comp = copy.copy(_canvas)
                    print(_comp.size)
                    #_comp = prev_comp_images[i]
                    #print(_comp)
                    _compmask = Image.new("RGBA", _comp.size, color='black')
                    _comp.paste(_layer, (0, 0))
                    _compmask.paste(_mask, (0, 0))
                    _compmask = _compmask.convert('L')
                    _comp = chop_image_v2(_canvas, _comp, "normal", 100)
                    #a_img = composite_masked(prev_comp_images[i], ret_images[i], mask=ret_masks[i])
                    print("TEST!!!!!")

                    # composition background
                    _canvas.paste(_comp, mask=_compmask)
                    #print(a_img)
                    comp_images.append(pil2tensor(_canvas))
                prev_comp_images = comp_images

            mask_count += 1

            prlx_x += x
            prlx_y += y
            prlx_zoom += zoom - 1

            temp_x = 0
            temp_y = 0
            temp_zoom = 0

        print(comp_images)
        return (torch.cat(comp_images, dim=0),)
        #return (comp_images,)



NODE_CLASS_MAPPINGS = {
    "ParallaxTest": parallax_test,
    "easy_parallax": easy_parallax,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParallaxTest": "Parallax Test",
    "EasyParallax": "Easy Parallax",
}