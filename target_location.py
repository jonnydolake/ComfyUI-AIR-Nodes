
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import math
import torch
import torchvision.transforms.functional as TF
import comfy
from comfy.utils import common_upscale
import numpy as np



#from masquerade nodes
def tensor2mask(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t
    if size[3] == 1:
        return t[:,:,:,0]
    elif size[3] == 4:
        # Not sure what the right thing to do here is. Going to try to be a little smart and use alpha unless all alpha is 1 in case we'll fallback to RGB behavior
        if torch.min(t[:, :, :, 3]).item() != 1.:
            return t[:,:,:,3]

    return TF.rgb_to_grayscale(tensor2rgb(t).permute(0,3,1,2), num_output_channels=1)[:,0,:,:]

#from masquerade nodes
def tensor2rgb(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 3)
    if size[3] == 1:
        return t.repeat(1, 1, 1, 3)
    elif size[3] == 4:
        return t[:, :, :, :3]
    else:
        return t


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# PIL to Mask
def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask

# Mask to PIL
def mask2pil(mask):
    if mask.ndim > 2:
        mask = mask.squeeze(0)
    mask_np = mask.cpu().numpy().astype('uint8')
    mask_pil = Image.fromarray(mask_np, mode="L")
    return mask_pil


def create_black(width, height):
    r = torch.full([1, height, width, 1], ((0 >> 16) & 0xFF) / 0xFF)
    g = torch.full([1, height, width, 1], ((0 >> 8) & 0xFF) / 0xFF)
    b = torch.full([1, height, width, 1], ((0) & 0xFF) / 0xFF)
    return torch.cat((r, g, b), dim=-1)

def create_white(width, height):
    r = torch.full([1, height, width, 1], ((16777215 >> 16) & 0xFF) / 0xFF)
    g = torch.full([1, height, width, 1], ((16777215 >> 8) & 0xFF) / 0xFF)
    b = torch.full([1, height, width, 1], ((16777215) & 0xFF) / 0xFF)
    return torch.cat((r, g, b), dim=-1)

#from impact pack
def batch_to_list(image):
    images = [image[i:i + 1, ...] for i in range(image.shape[0])]
    return images

#from impact pack
def list_to_batch(images):
    if len(images) <= 1:
        return images[0]
    else:
        image1 = images[0]
        for image2 in images[1:]:
            if image1.shape[1:] != image2.shape[1:]:
                image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "lanczos", "center").movedim(1, -1)
            image1 = torch.cat((image1, image2), dim=0)
        return image1

def mask_to_image(mask):
    result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    return result

def image_to_mask(image):
    if len(image.shape) > 3 and image.shape[3] == 4:
        image = tensor2rgb(image)
    return tensor2mask(image)

def make_3d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0)

    elif len(mask.shape) == 2:
        return mask.unsqueeze(0)

    return mask

def mask_to_list(masks):
    if masks is None:
        empty_mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return ([empty_mask], )

    res = []

    for mask in masks:
        res.append(mask)

    #print(f"mask len: {len(res)}")

    res = [make_3d_mask(x) for x in res]

    return res

def crop_region(mask, padding=12):
    from scipy.ndimage import label, find_objects
    binary_mask = np.array(mask.convert("L")) > 0
    bbox = mask.getbbox()
    if bbox is None:
        return mask, (mask.size, (0, 0, 0, 0))

    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]

    side_length = max(bbox_width, bbox_height) + 2 * padding

    center_x = (bbox[2] + bbox[0]) // 2
    center_y = (bbox[3] + bbox[1]) // 2

    crop_x = center_x - side_length // 2
    crop_y = center_y - side_length // 2

    crop_x = max(crop_x, 0)
    crop_y = max(crop_y, 0)
    crop_x2 = min(crop_x + side_length, mask.width)
    crop_y2 = min(crop_y + side_length, mask.height)

    cropped_mask = mask.crop((crop_x, crop_y, crop_x2, crop_y2))
    crop_data = (cropped_mask.size, (crop_x, crop_y, crop_x2, crop_y2))

    return cropped_mask, crop_data

def new_mask_crop_region(mask, padding=12):
    region_mask, crop_data = crop_region(mask.convert('L'), padding)
    (width, height), (left, top, right, bottom) = crop_data

    return ImageOps.invert(region_mask), top, left, right, bottom

def new_image_crop_location(image, top=0, left=0, right=256, bottom=256):
    img_width, img_height = image.size

    # Calculate the final coordinates for cropping
    crop_top = max(top, 0)
    crop_left = max(left, 0)
    crop_bottom = min(bottom, img_height)
    crop_right = min(right, img_width)

    # Ensure that the cropping region has non-zero width and height
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("Invalid crop dimensions. Please check the values for top, left, right, and bottom.")

    # Crop the image and resize
    crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    #crop_data = (crop.size, (crop_left, crop_top, crop_right, crop_bottom))
    crop = crop.resize((((crop.size[0] // 8) * 8), ((crop.size[1] // 8) * 8)))

    return crop

def inset_border(image, border_width=20, border_color=(0)):
    width, height = image.size
    bordered_image = Image.new(image.mode, (width, height), border_color)
    bordered_image.paste(image, (0, 0))
    draw = ImageDraw.Draw(bordered_image)
    draw.rectangle((0, 0, width-1, height-1), outline=border_color, width=border_width)
    return bordered_image


def new_paste_image(image, crop_image, top=0, left=0, right=256, bottom=256, blend_amount=0.25):

    image = image.convert("RGBA")
    crop_image = crop_image.convert("RGBA")

    img_width, img_height = image.size

    # Ensure that the coordinates are within the image bounds
    top = min(max(top, 0), img_height)
    left = min(max(left, 0), img_width)
    bottom = min(max(bottom, 0), img_height)
    right = min(max(right, 0), img_width)

    crop_size = (right - left, bottom - top)
    crop_img = crop_image.resize(crop_size)
    crop_img = crop_img.convert("RGBA")

    if blend_amount > 1.0:
        blend_amount = 1.0
    elif blend_amount < 0.0:
        blend_amount = 0.0
    blend_ratio = (max(crop_size) / 2) * float(blend_amount)

    blend = image.copy()
    mask = Image.new("L", image.size, 0)

    mask_block = Image.new("L", crop_size, 255)
    mask_block = inset_border(mask_block, int(blend_ratio/2), (0))

    Image.Image.paste(mask, mask_block, (left, top))
    blend.paste(crop_img, (left, top), crop_img)

    mask = mask.filter(ImageFilter.BoxBlur(radius=blend_ratio/4))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_ratio/4))

    blend.putalpha(mask)
    image = Image.alpha_composite(image, blend)

    return image.convert('RGB')


def composite(destination, source, x, y, mask = None, multiplier = 8, resize_source = False):
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
    y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[3], top + source.shape[2],)

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    # calculate the bounds of the source that will be overlapping the destination
    # this prevents the source trying to overwrite latent pixels that are out of bounds
    # of the destination
    visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

    mask = mask[:, :, :visible_height, :visible_width]
    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[:, :, :visible_height, :visible_width]
    destination_portion = inverse_mask  * destination[:, :, top:bottom, left:right]

    destination[:, :, top:bottom, left:right] = source_portion + destination_portion
    return destination

def composite_masked(destination, source, x=0, y=0, resize_source = False, mask = None):
    destination = destination.clone().movedim(-1, 1)
    output = composite(destination, source.movedim(-1, 1), x, y, mask, 1, resize_source).movedim(1, -1)
    return output

#from derfuu nodes
def get_image_size(IMAGE) -> tuple[int, int]:
    samples = IMAGE.movedim(-1, 1)
    size = samples.shape[3], samples.shape[2]
    # size = size.movedim(1, -1)
    return size


def new_upscale(image, side_length: int):

    size = image.size

    image_width = int(size[0])
    image_height = int(size[1])

    width = image_width
    height = image_height

    def determineSide(_side: str) -> tuple[int, int]:
        width, height = 0, 0
        if _side == "Width":
            height_ratio = image_height / image_width
            width = side_length
            height = height_ratio * width
        elif _side == "Height":
            width_ratio = image_width / image_height
            height = side_length
            width = width_ratio * height
        return width, height


    if width > height:
        width, height = determineSide("Width")
    else:
        width, height = determineSide("Height")

    width = math.ceil(width)
    height = math.ceil(height)

    new_size = (width, height)

    resized_image = image.resize(new_size, resample=Image.Resampling.LANCZOS)
    return resized_image


#tensor tests
def tensor_crop_region(mask_tensor, padding=12):
    # Convert mask tensor to numpy for region detection
    mask_np = mask_tensor.squeeze().cpu().numpy() > 0.5

    # Find bounding box
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, mask_np.shape[0])
    xmin, xmax = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, mask_np.shape[1])

    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    side_length = max(bbox_width, bbox_height) + 2 * padding

    center_x = (xmax + xmin) // 2
    center_y = (ymax + ymin) // 2

    crop_x = center_x - side_length // 2
    crop_y = center_y - side_length // 2

    crop_x = max(crop_x, 0)
    crop_y = max(crop_y, 0)
    crop_x2 = min(crop_x + side_length, mask_np.shape[1])
    crop_y2 = min(crop_y + side_length, mask_np.shape[0])

    return crop_x, crop_y, crop_x2, crop_y2


def tensor_upscale(image_tensor, side_length):
    _, height, width, _ = image_tensor.shape

    if width > height:
        new_width = side_length
        new_height = int(height * (side_length / width))
    else:
        new_height = side_length
        new_width = int(width * (side_length / height))

    # Use torch.nn.functional.interpolate
    resized = torch.nn.functional.interpolate(
        image_tensor.permute(0, 3, 1, 2),
        size=(new_height, new_width),
        mode='bilinear',
        align_corners=False,
        antialias=True
    ).permute(0, 2, 3, 1)

    return resized


def tensor_paste(dest_tensor, src_tensor, x, y, mask_tensor=None, blend_amount=0.0):
    if mask_tensor is None:
        mask_tensor = torch.ones_like(src_tensor[:, :, :, :1])

    # Calculate blend region
    blend_radius = int((max(src_tensor.shape[1], src_tensor.shape[2]) / 2) * blend_amount)

    # Create blend mask
    if blend_radius > 0:
        # Create distance mask
        h, w = src_tensor.shape[1], src_tensor.shape[2]
        y_coords = torch.arange(h, device=src_tensor.device).float() - h / 2
        x_coords = torch.arange(w, device=src_tensor.device).float() - w / 2
        y_dist = y_coords.unsqueeze(1).repeat(1, w)
        x_dist = x_coords.unsqueeze(0).repeat(h, 1)
        dist = torch.sqrt(x_dist ** 2 + y_dist ** 2)
        max_dist = torch.max(dist)
        blend_mask = 1 - torch.clamp((dist - (max_dist - blend_radius)) / blend_radius, 0, 1)
        blend_mask = blend_mask.unsqueeze(0).unsqueeze(-1)
    else:
        blend_mask = torch.ones_like(src_tensor[:, :, :, :1])

    # Combine with input mask
    final_mask = mask_tensor * blend_mask

    # Calculate paste coordinates
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + src_tensor.shape[2], dest_tensor.shape[2]), min(y + src_tensor.shape[1], dest_tensor.shape[1])

    src_x1, src_y1 = max(-x, 0), max(-y, 0)
    src_x2, src_y2 = src_x1 + (x2 - x1), src_y1 + (y2 - y1)

    # Perform the paste operation
    dest_slice = dest_tensor[:, y1:y2, x1:x2, :]
    src_slice = src_tensor[:, src_y1:src_y2, src_x1:src_x2, :]
    mask_slice = final_mask[:, src_y1:src_y2, src_x1:src_x2, :]

    dest_tensor[:, y1:y2, x1:x2, :] = dest_slice * (1 - mask_slice) + src_slice * mask_slice

    return dest_tensor






class target_location_crop:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images" :("IMAGE",),
                "masks": ("MASK",),
                "area_mode": ("BOOLEAN", {"default": False}),
                "resolution_size": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 64, "display": "number", "lazy": True}),
                "mask_padding": ("INT", {"default": 12, "min": 0, "max": 128, "step": 1, "display": "number", "lazy": True}),
                "invert_masks": ("BOOLEAN", { "default": False })
            }
        }

    RETURN_TYPES = ("IMAGE","MASK","*")
    RETURN_NAMES = ("cropped_images","cropped_masks", "crop_data")
    #OUTPUT_IS_LIST = (False, True, False)
    FUNCTION = "crop"

    CATEGORY = "AIR Nodes"

    def crop(self, images, masks, resolution_size, mask_padding, invert_masks, area_mode):

        if range(len(images)) != range(len(masks)):
            raise ValueError("Image count[" + str(len(images)) + "] and Mask count["+ str(len(masks)) + "] need to be the same")

        padding = mask_padding
        size = resolution_size

        if invert_masks:
            masks = 1.0 - masks


        _images = []
        _masks = []

        output_images = []
        cropped_images = []
        resized_cropped_images = []
        cropped_images_scaled = []
        cropped_masks = []
        cropped_masks_scaled = []
        top = []
        left = []
        right = []
        bottom = []

        for x in range(len(images)):
            _images.append(tensor2pil(images[x]))
            _masks.append(tensor2pil(masks[x]))



        if area_mode:

            for _mask in _masks:
                x_cropped_mask, x_top, x_left, x_right, x_bottom = new_mask_crop_region(_mask, padding)
                top.append(x_top)
                left.append(x_left)
                right.append(x_right)
                bottom.append(x_bottom)

            for _image in _images:
                cropped_image = new_image_crop_location(_image, min(top), min(left), max(right), max(bottom))
                cropped_images.append(cropped_image)
                cropped_images_scaled.append(pil2tensor(new_upscale(cropped_image, size)))

            for _mask in _masks:
                cropped_mask = new_image_crop_location(_mask, min(top), min(left), max(right), max(bottom))
                cropped_masks.append(cropped_mask.convert("L"))
                #color_mask_tensor = torch.tensor(cropped_mask, dtype=torch.float32)
                cropped_masks_scaled.append(pil2tensor(new_upscale(cropped_mask, size)))

            coordinates = (_images, cropped_images, _masks, min(top), min(left), max(right), max(bottom), area_mode)

            return (torch.cat(cropped_images_scaled, dim=0), image_to_mask(list_to_batch(cropped_masks_scaled)), coordinates)


        for x in range(len(images)):
            _mask = tensor2pil(masks[x])
            _image = tensor2pil(images[x])
            black_comp = Image.new('L', size=[size, size], color='black')
            white_comp = Image.new('RGB', size=[size, size], color='white')

            x_cropped_mask, x_top, x_left, x_right, x_bottom = new_mask_crop_region(_mask, padding)
            cropped_image = new_image_crop_location(_image, x_top, x_left, x_right, x_bottom)
            cropped_masks.append(x_cropped_mask)
            top.append(x_top)
            left.append(x_left)
            right.append(x_right)
            bottom.append(x_bottom)

            cropped_images.append(cropped_image)
            resized_cropped_images.append(new_upscale(cropped_image, size))
            white_comp.paste(new_upscale(cropped_image, size), (0, 0))
            cropped_images_scaled.append(white_comp)

            black_comp.paste(new_upscale(x_cropped_mask, size), (0, 0))
            cropped_masks_scaled.append(pil2tensor(black_comp))
            output_images.append(pil2tensor(white_comp))

        coordinates = (_images, cropped_images_scaled, cropped_images, _masks, top, left, right, bottom, resized_cropped_images, area_mode)

        return (torch.cat(output_images, dim=0), torch.cat(cropped_masks_scaled, dim=0), coordinates)

class target_location_paste:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cropped_images" :("IMAGE",),
                #"masks": ("MASK",)
                "crop_data": ("*", {"forceInput": True}),
                "apply_masks": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    #OUTPUT_IS_LIST = (True,)
    FUNCTION = "paste"

    CATEGORY = "AIR Nodes"

    def paste(self, cropped_images, crop_data, apply_masks):

        area_mode = crop_data[-1]

        _images = []

        for img in cropped_images:
            _images.append(tensor2pil(img))


        if area_mode:

            original_images = crop_data[0]


            original_cropped_images = crop_data[1]
            original_masks = crop_data[2]
            original_width, original_height = original_images[0].size
            original_empty_image = Image.new('RGB', size=[int(original_width), int(original_height)], color='white')

            top = crop_data[3]
            left = crop_data[4]
            right = crop_data[5]
            bottom = crop_data[6]


            final_images = []

            for x in range(len(_images)):
                if _images[x].size > original_cropped_images[0].size or _images[x].size < original_cropped_images[0].size:
                    image_size = original_cropped_images[0].size


                    temp1 = _images[x]

                    temp_image = temp1.resize(image_size, resample=Image.Resampling.LANCZOS)
                else:
                    temp_image = _images[x]


                if apply_masks and len(_images) != len(original_masks):
                    raise AttributeError(
                        f"Image size is {str(len(cropped_images))} while Mask size is {str(len(original_masks))}! \n Either match the Image size and Mask size or set 'apply_masks' to False!")

                elif len(_images) == len(original_masks):
                    comp_image = new_paste_image(original_images[x], temp_image, top, left, right, bottom, 0.0)
                else:
                    if x == 0:
                        print("\033[31mNOTE: Your cropped images will get pasted on an empty image because their count exceeds the count of the original batch!\033[0m")

                    comp_image = new_paste_image(original_empty_image, temp_image, top, left, right, bottom, 0.0)


                if apply_masks and len(_images) == len(original_masks):
                    final_image = Image.composite(original_images[x], comp_image, ImageOps.invert(original_masks[x].convert('L')))

                else:
                    final_image = comp_image


                final_images.append(pil2tensor(final_image))

            return (torch.cat(final_images, dim=0),)

        original_images = crop_data[0]
        resized_cropped_images = crop_data[8]
        cropped_images_scaled = crop_data[1]
        cropped_image_list = crop_data[2]
        original_masks = crop_data[3]

        top = crop_data[4]
        left = crop_data[5]
        right = crop_data[6]
        bottom = crop_data[7]

        final_images= []

        for x in range(len(_images)):
            if cropped_images_scaled[x].size < _images[x].size or cropped_images_scaled[x].size > _images[x].size:

                image_width, image_height = cropped_images_scaled[x].size

                temp_image_resized = new_upscale(_images[x], image_width)

                resized_cropped_images[x].paste(temp_image_resized, (0,0))


            else:
                resized_cropped_images[x].paste(_images[x], (0, 0))


            new_image_list_resized = resized_cropped_images[x].resize(cropped_image_list[x].size, resample=Image.Resampling.LANCZOS)

            comp_image = new_paste_image(original_images[x], new_image_list_resized, top[x], left[x], right[x], bottom[x], 0.0)

            if apply_masks:
                final_image = Image.composite(original_images[x], comp_image, ImageOps.invert(original_masks[x].convert('L')))

            else:
                final_image = comp_image

            final_images.append(pil2tensor(final_image))

        return (torch.cat(final_images, dim=0),)


class image_composite_chained:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "main_destination" :("IMAGE", ),
                "sources": ("IMAGE", ),
                "masks": ("MASK", ),
                "composite_chained": ("BOOLEAN", {"default": True}),
                "destination": (["main_destination", "empty_white_image", "empty_black_image"],),
                "resize_sources": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "run"

    CATEGORY = "AIR Nodes"

    def run(self, main_destination, sources, masks, composite_chained, destination, resize_sources):

        if destination == "empty_white_image":
            size = get_image_size(main_destination)
            width_B = int(size[0])
            height_B = int(size[1])
            composited_image = create_white(width_B,height_B)
        elif destination == "empty_black_image":
            size = get_image_size(main_destination)
            width_B = int(size[0])
            height_B = int(size[1])
            composited_image = create_black(width_B,height_B)
        else:
            composited_image = main_destination

        source_list = batch_to_list(sources)
        mask_list = mask_to_list(masks)

        if composite_chained:
            for x in range(len(source_list)):
                composited_image = composite_masked(composited_image, source_list[x], resize_source=resize_sources, mask=mask_list[x])

            return (composited_image,)
        else:
            composited_images = []
            for x in range(len(source_list)):
                new_composited_image = composite_masked(composited_image, source_list[x], resize_source=resize_sources, mask=mask_list[x])
                composited_images.append(new_composited_image)
            return (list_to_batch(composited_images),)


class tensor_target_location_crop:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "area_mode": ("BOOLEAN", {"default": False}),
                "resolution_size": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 64}),
                "mask_padding": ("INT", {"default": 12, "min": 0, "max": 128, "step": 1}),
                "invert_masks": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "*", "INT")
    RETURN_NAMES = ("cropped_images", "cropped_masks", "crop_data", "original_crop_size")
    FUNCTION = "crop"
    CATEGORY = "AIR Nodes"

    def crop(self, images, masks, resolution_size, mask_padding, invert_masks, area_mode):
        if len(images) != len(masks):
            raise ValueError(f"Image count[{len(images)}] and Mask count[{len(masks)}] need to be the same")

        if invert_masks:
            masks = 1.0 - masks

        device = images.device
        padding = mask_padding
        size = resolution_size
        white_square = create_white(resolution_size, resolution_size)
        black_square = create_black(resolution_size, resolution_size)

        # Convert masks to single channel if needed
        if masks.ndim == 4 and masks.shape[3] > 1:
            masks = masks.mean(dim=-1, keepdim=True)

        batch_size = images.shape[0]
        cropped_images = []
        cropped_masks = []
        crop_data = []

        _images = []
        _masks = []

        original_crop_size = 0

        for x in range(len(images)):
            _images.append(tensor2pil(images[x]))
            _masks.append(tensor2pil(masks[x]))

        if area_mode:
            # Find union of all mask regions
            top, left, right, bottom = [], [], [], []
            for _mask in _masks:
                x_cropped_mask, x_top, x_left, x_right, x_bottom = new_mask_crop_region(_mask, padding)
                top.append(x_top)
                left.append(x_left)
                right.append(x_right)
                bottom.append(x_bottom)

            for _image in _images:
                cropped_image = new_image_crop_location(_image, min(top), min(left), max(right), max(bottom))

                #get original crop size (we only need the highest value)
                width, height = cropped_image.size
                if width > height:
                    original_crop_size = width
                else:
                    original_crop_size = height

                cropped_images.append(pil2tensor(new_upscale(cropped_image, size)))

            for _mask in _masks:
                cropped_mask = new_image_crop_location(_mask, min(top), min(left), max(right), max(bottom))
                cropped_masks.append(pil2tensor(new_upscale(cropped_mask, size)))

            crop_data.append((min(left), min(top), max(right), max(bottom)))


            crop_data_tuple = (images, cropped_images, masks, crop_data, True)
            return (torch.cat(cropped_images, dim=0), torch.cat(cropped_masks, dim=0), crop_data_tuple, original_crop_size)

        else:
            original_crop = []
            original_crop_masks = []
            # Process each image/mask pair individually
            for i in range(batch_size):
                mask_tensor = masks[i].unsqueeze(0)
                x1, y1, x2, y2 = tensor_crop_region(mask_tensor, padding)

                # Crop and resize image
                img_crop = images[i:i + 1, y1:y2, x1:x2, :]
                img_crop = tensor_upscale(img_crop, size)
                original_crop.append(img_crop)
                img_crop_scaled = composite_masked(white_square, img_crop)
                cropped_images.append(img_crop_scaled)

                # Crop and resize mask
                mask_crop = masks[i:i + 1, y1:y2, x1:x2]
                mask_crop = tensor_upscale(mask_crop.unsqueeze(-1), size)
                original_crop_masks.append(mask_crop.squeeze(-1))
                mask_crop_scaled = composite_masked(black_square, mask_crop)
                cropped_masks.append(image_to_mask(mask_crop_scaled))

                # Store crop data
                crop_data.append((x1, y1, x2, y2))

            crop_data_tuple = (images, cropped_masks, get_image_size(cropped_images[0]), original_crop, crop_data, False)
            return (torch.cat(cropped_images, dim=0), torch.cat(cropped_masks, dim=0), crop_data_tuple)


class tensor_target_location_paste:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cropped_images": ("IMAGE",),
                "crop_data": ("*", {"forceInput": True}),
                "apply_masks": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "paste"
    CATEGORY = "AIR Nodes"

    def paste(self, cropped_images, crop_data, apply_masks):
        area_mode = crop_data[-1]
        if area_mode:
            original_images, original_cropped_images, original_masks, crop_info, area_mode = crop_data
        else:
            original_images, original_crop_masks, crop_size, original_crop, crop_info, area_mode = crop_data

        device = original_images.device
        batch_size = cropped_images.shape[0]
        output_images = []

        # Get original image dimensions
        _, original_height, original_width, _ = original_images.shape

        if area_mode:
            # Area mode - all images use the same crop coordinates
            x1, y1, x2, y2 = crop_info[0]
            crop_height = y2 - y1
            crop_width = x2 - x1

            # Determine if we should use blank canvas for all
            use_blank_canvas = (batch_size > len(original_cropped_images))

            for i in range(batch_size):
                # Always use blank canvas if we have more cropped images than originals
                if use_blank_canvas:
                    canvas = torch.ones((1, original_height, original_width, 3), device=device)
                else:
                    canvas = original_images[i:i + 1].clone()

                # Resize cropped image to original crop size
                resized_crop = torch.nn.functional.interpolate(
                    cropped_images[i:i + 1].permute(0, 3, 1, 2),
                    size=(crop_height, crop_width),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True
                ).permute(0, 2, 3, 1)


                # Paste onto canvas
                canvas[:, y1:y2, x1:x2, :] = resized_crop

                # Apply mask if not using blank canvas and masks are enabled
                if not use_blank_canvas and apply_masks and i < len(original_masks):
                    mask = original_masks[i:i + 1].unsqueeze(-1)
                    #canvas = canvas * (1 - mask) + resized_crop * mask
                    canvas = composite_masked(original_images[i:i + 1], canvas, mask=mask)

                output_images.append(canvas)

            return (torch.cat(output_images, dim=0),)

        else:
            # Individual mode - original behavior
            for i in range(batch_size):
                if i >= len(crop_info):
                    break  # Safety check

                if get_image_size(cropped_images[i:i + 1]) != crop_size:
                    print('True')
                    width = int(crop_size[0])
                    height = int(crop_size[1])
                    scaled_tensor = torch.nn.functional.interpolate(
                        cropped_images[i:i + 1].permute(0, 3, 1, 2),
                        size=(height, width),
                        mode='bilinear',
                        align_corners=False,
                        antialias=True
                    ).permute(0, 2, 3, 1)
                else:
                    scaled_tensor = cropped_images[i:i + 1]

                # Apply mask if requested
                if apply_masks:
                    comp_image = composite_masked(original_crop[i], scaled_tensor, mask=original_crop_masks[i])
                else:
                    comp_image = composite_masked(original_crop[i], scaled_tensor)

                x1, y1, x2, y2 = crop_info[i]
                crop_height = y2 - y1
                crop_width = x2 - x1

                # Resize cropped image to original crop size
                resized_crop = torch.nn.functional.interpolate(
                    comp_image.permute(0, 3, 1, 2),
                    size=(crop_height, crop_width),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True
                ).permute(0, 2, 3, 1)

                # Paste onto original image
                output = original_images[i:i + 1].clone()
                output[:, y1:y2, x1:x2, :] = resized_crop

                # Apply mask if requested
                '''if apply_masks and i < len(original_masks):
                    mask = original_masks[i:i + 1].unsqueeze(-1)
                    output = output * (1 - mask) + resized_crop * mask'''

                output_images.append(output)

            return (torch.cat(output_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "TargetLocationCrop": target_location_crop,
    "TargetLocationPaste": target_location_paste,
    "GPUTargetLocationCrop": tensor_target_location_crop,
    "GPUTargetLocationPaste": tensor_target_location_paste,
    "ImageCompositeChained": image_composite_chained,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TargetLocationCrop": "Target Location (Crop)",
    "TargetLocationPaste": "Target Location (Paste)",
    "GPUTargetLocationCrop": "GPU Target Location Crop",
    "GPUTargetLocationPaste": "GPU Target Location Paste",
    "ImageCompositeChained": "Image Composite Chained",

}
