import torch
from PIL import Image
import numpy as np

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def RGB2RGBA(image: Image, mask: Image) -> Image:
    (R, G, B) = image.convert('RGB').split()
    return Image.merge('RGBA', (R, G, B, mask.convert('L')))

class ParallaxGPUTest:
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
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'layer_image_transform'
    CATEGORY = 'AIR Nodes'

    def layer_image_transform(self, image, frames, x, y, zoom, aspect_ratio):
        device = image.device
        l_images = []
        l_masks = []
        prev_comp_images = []
        comp_images = []

        # Pre-allocate lists for better memory efficiency
        for l in image:
            # Move tensor to GPU and keep it there
            l = l.to(device)
            temp_img_list = []
            temp_mask_list = []
            
            # Create frame copies efficiently
            frame_tensor = torch.unsqueeze(l, 0)
            for _ in range(frames):
                temp_img_list.append(frame_tensor)
                
                # Extract alpha if RGBA
                pil_img = tensor2pil(l)
                if pil_img.mode == 'RGBA':
                    alpha = pil_img.split()[-1]
                    temp_mask_list.append(alpha)
            
            l_images.append(temp_img_list)
            l_masks.append(temp_mask_list)

        mask_count = 0
        temp_x = 0
        temp_y = 0
        temp_zoom = 1.00
        prlx_x = 0
        prlx_y = 0
        prlx_zoom = 0.00

        # Process layers
        for layer_idx, img in enumerate(l_images):
            temp_x += prlx_x
            temp_y += prlx_y
            temp_zoom += prlx_zoom

            ret_images = []
            ret_masks = []

            # Process frames
            for i in range(len(img)):
                if i == 0:
                    temp_x = 0
                    temp_y = 0
                    temp_zoom = 1.00

                # Keep tensors on GPU
                layer_image = img[i].to(device) if i < len(img) else img[-1].to(device)
                _image = tensor2pil(layer_image).convert('RGB')

                # Handle masks
                if i < len(l_masks[mask_count]):
                    _mask = l_masks[mask_count][i]
                else:
                    _mask = Image.new('L', size=_image.size, color='white')

                # Create canvases
                _image_canvas = Image.new('RGB', size=_image.size, color='black')
                _mask_canvas = Image.new('L', size=_mask.size, color='black')

                # Calculate dimensions
                orig_layer_width = _image.width
                orig_layer_height = _image.height
                target_layer_width = int(orig_layer_width * temp_zoom)
                target_layer_height = int(orig_layer_height * temp_zoom * aspect_ratio)

                # Resize
                _image = _image.resize((target_layer_width, target_layer_height))
                _mask = _mask.resize((target_layer_width, target_layer_height))

                # Calculate paste coordinates
                paste_x = (orig_layer_width - target_layer_width) // 2 + temp_x
                paste_y = (orig_layer_height - target_layer_height) // 2 + temp_y

                # Paste images
                _image_canvas.paste(_image, (paste_x, paste_y))
                _mask_canvas.paste(_mask, (paste_x, paste_y))

                if tensor2pil(layer_image).mode == 'RGBA':
                    _image_canvas = RGB2RGBA(_image_canvas, _mask_canvas)

                # Convert to tensor and move to GPU
                ret_images.append(pil2tensor(_image_canvas).to(device))
                ret_masks.append(pil2tensor(_mask_canvas).to(device))

                temp_x += x + prlx_x
                temp_y += y + prlx_y
                temp_zoom += zoom - 1 + prlx_zoom

            # Layer composition
            if mask_count == 0:
                prev_comp_images = ret_images
                comp_images = ret_images
            else:
                comp_images = []
                for i in range(len(prev_comp_images)):
                    _canvas = tensor2pil(prev_comp_images[i]).convert('RGBA')
                    _layer = tensor2pil(ret_images[i])
                    _mask = tensor2pil(ret_masks[i])

                    _comp = Image.new("RGBA", _canvas.size, (0, 0, 0, 0))
                    _comp.paste(_canvas, (0, 0))
                    _comp.paste(_layer, (0, 0), mask=_mask)
                    
                    # Move composed image to GPU
                    comp_images.append(pil2tensor(_comp).to(device))

                prev_comp_images = comp_images

            mask_count += 1

            prlx_x += x
            prlx_y += y
            prlx_zoom += zoom - 1

            temp_x = 0
            temp_y = 0
            temp_zoom = 0

        # Final concatenation on GPU
        result = torch.cat(comp_images, dim=0)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "ParallaxGPUTest": ParallaxGPUTest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParallaxGPUTest": "Parallax GPU Test",
}