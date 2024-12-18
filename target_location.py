
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

    print(f"mask len: {len(res)}")

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

def mask_crop_region(mask, padding=12):

    mask_pil = Image.fromarray(np.clip(255. * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    region_mask, crop_data = crop_region(mask_pil, padding)
    region_tensor = pil2mask(ImageOps.invert(region_mask)).unsqueeze(0).unsqueeze(1)

    (width, height), (left, top, right, bottom) = crop_data

    return region_tensor, top, left, right, bottom

def image_crop_location(image, top=0, left=0, right=256, bottom=256):
    image = tensor2pil(image)
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

    return pil2tensor(crop)

def paste_image(image, crop_image, top=0, left=0, right=256, bottom=256, blend_amount=0.25, sharpen_amount=1):

    image = image.convert("RGBA")
    crop_image = crop_image.convert("RGBA")

    def inset_border(image, border_width=20, border_color=(0)):
        width, height = image.size
        bordered_image = Image.new(image.mode, (width, height), border_color)
        bordered_image.paste(image, (0, 0))
        draw = ImageDraw.Draw(bordered_image)
        draw.rectangle((0, 0, width-1, height-1), outline=border_color, width=border_width)
        return bordered_image

    img_width, img_height = image.size

    # Ensure that the coordinates are within the image bounds
    top = min(max(top, 0), img_height)
    left = min(max(left, 0), img_width)
    bottom = min(max(bottom, 0), img_height)
    right = min(max(right, 0), img_width)

    crop_size = (right - left, bottom - top)
    crop_img = crop_image.resize(crop_size)
    crop_img = crop_img.convert("RGBA")

    if sharpen_amount > 0:
        for _ in range(sharpen_amount):
            crop_img = crop_img.filter(ImageFilter.SHARPEN)

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

    return (pil2tensor(image), pil2tensor(mask.convert('RGB')))

def image_paste_crop_location(image, crop_image, top=0, left=0, right=256, bottom=256, crop_blending=0, crop_sharpening=0):
    result_image, result_mask = paste_image(tensor2pil(image), tensor2pil(crop_image), top, left, right, bottom, crop_blending, crop_sharpening)
    return result_image

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

#from derfuu nodes
def upscale(image, side_length: int):
    samples = image.movedim(-1, 1)

    size = get_image_size(image)

    width_B = int(size[0])
    height_B = int(size[1])

    width = width_B
    height = height_B

    def determineSide(_side: str) -> tuple[int, int]:
        width, height = 0, 0
        if _side == "Width":
            heigh_ratio = height_B / width_B
            width = side_length
            height = heigh_ratio * width
        elif _side == "Height":
            width_ratio = width_B / height_B
            height = side_length
            width = width_ratio * height
        return width, height


    if width > height:
        width, height = determineSide("Width")
    else:
        width, height = determineSide("Height")

    width = math.ceil(width)
    height = math.ceil(height)

    cls = common_upscale(samples, width, height, "lanczos", "disabled")
    cls = cls.movedim(1, -1)
    return cls

def image_to_rgb(images):

    if len(images) > 1:
        tensors = []
        for image in images:
            tensors.append(pil2tensor(tensor2pil(image).convert('RGB')))
        tensors = torch.cat(tensors, dim=0)
        return (tensors, )
    else:
        return pil2tensor(tensor2pil(images).convert("RGB"))

class target_location_crop:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images" :("IMAGE",),
                "masks": ("IMAGE",),
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
        padding = mask_padding
        size = resolution_size

        if invert_masks == True:
            masks = 1.0 - masks

        black_square = create_black(size, size)
        white_square = create_white(size, size)

        converted_masks = image_to_mask(masks)

        if area_mode:
            cropped_images = []
            cropped_images_scaled = []
            cropped_masks = []
            cropped_masks_scaled = []
            top = []
            left = []
            right = []
            bottom = []

            for x in range(len(converted_masks)):
                x_cropped_mask, x_top, x_left, x_right, x_bottom = mask_crop_region(converted_masks[x], padding)
                top.append(x_top)
                left.append(x_left)
                right.append(x_right)
                bottom.append(x_bottom)

            for x in range(len(images)):
                cropped_image = image_crop_location(images[x], min(top), min(left), max(right), max(bottom))
                cropped_images.append(cropped_image)
                cropped_images_scaled.append(upscale(cropped_image, size))

            for x in range(len(masks)):
                cropped_mask = image_crop_location(masks[x], min(top), min(left), max(right), max(bottom))
                cropped_masks.append(cropped_mask)
                cropped_masks_scaled.append(upscale(cropped_mask, size))

            coordinates = (images, cropped_images, cropped_masks, min(top), min(left), max(right), max(bottom), area_mode)

            return (list_to_batch(cropped_images_scaled), image_to_mask(list_to_batch(cropped_masks_scaled)), coordinates)

        output_images = []
        cropped_images = []
        cropped_images_scaled = []
        cropped_masks = []
        cropped_masks_scaled = []
        top = []
        left = []
        right = []
        bottom = []
        test1 = range(len(converted_masks))
        for x in test1:
            x_cropped_mask, x_top, x_left, x_right, x_bottom = mask_crop_region(converted_masks[x], padding)
            cropped_image = image_crop_location(images[x], x_top, x_left, x_right, x_bottom)
            cropped_masks.append(x_cropped_mask)
            top.append(x_top)
            left.append(x_left)
            right.append(x_right)
            bottom.append(x_bottom)
            cropped_images.append(cropped_image)
            cropped_images_scaled.append(upscale(cropped_image, size))
            cropped_masks_scaled.append(composite_masked(black_square, upscale(mask_to_image(x_cropped_mask), size)))
            output_images.append(composite_masked(white_square,upscale(cropped_image, size)))

        coordinates = (images, cropped_images_scaled, cropped_images, cropped_masks, top, left, right, bottom, output_images, area_mode)

        return (list_to_batch(output_images), image_to_mask(list_to_batch(cropped_masks_scaled)), coordinates)

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

        if area_mode:
            original_images = batch_to_list(crop_data[0])
            new_cropped_images = batch_to_list(cropped_images)
            original_cropped_images = crop_data[1]
            cropped_masks = crop_data[2]
            original_image_size = original_images[0].size()
            original_empty_image = create_white(int(original_image_size[2]), int(original_image_size[1]))

            top = crop_data[3]
            left = crop_data[4]
            right = crop_data[5]
            bottom = crop_data[6]


            image_list = []

            for x in range(len(new_cropped_images)):
                if new_cropped_images[x].size() > original_cropped_images[0].size() or new_cropped_images[x].size() < original_cropped_images[0].size():
                    image_size = original_cropped_images[0].size()
                    image_width = int(image_size[2])
                    image_height = int(image_size[1])

                    temp1 = new_cropped_images[x]

                    samples = temp1.movedim(-1, 1)
                    temp_image = common_upscale(samples, image_width, image_height, "lanczos", "disabled")
                    temp_image = temp_image.movedim(1, -1)
                else:
                    temp_image = new_cropped_images[x]

                if apply_masks and len(new_cropped_images) == len(cropped_masks):
                    temp_image = composite_masked(original_cropped_images[x], temp_image, mask=image_to_mask(cropped_masks[x]))
                elif apply_masks and len(new_cropped_images) != len(cropped_masks):
                    raise AttributeError(
                        f"Image size is {str(len(new_cropped_images))} while Mask size is {str(len(cropped_masks))}! \n Either match the Image size and Mask size or set 'apply_masks' to False!")

                if len(new_cropped_images) == len(cropped_masks):
                    print("YES")
                    image_list_rgba = image_paste_crop_location(original_images[x], temp_image, top, left, right, bottom)
                else:
                    if x == 0:
                        print("\033[31mNOTE: Your cropped images will get pasted on an empty image because their count exceeds the count of the original batch!\033[0m")

                    image_list_rgba = image_paste_crop_location(original_empty_image, temp_image, top, left, right, bottom)

                image_list.append(image_to_rgb(image_list_rgba))

            return (list_to_batch(image_list),)

        original_images = batch_to_list(crop_data[0])
        images= batch_to_list(cropped_images)
        cropped_images_scaled = crop_data[1]
        cropped_image_list = crop_data[2]
        cropped_masks = crop_data[3]
        original_output_images = crop_data[8]

        top = crop_data[4]
        left = crop_data[5]
        right = crop_data[6]
        bottom = crop_data[7]

        image_list= []

        for x in range(len(images)):
            if original_output_images[x].size() < images[x].size() or original_output_images[x].size() > images[x].size():
                image_size = images[x].size()
                image_width = int(image_size[2])

                temp_image_list_resized = upscale(cropped_images_scaled[x], image_width)

                new_image_list = composite_masked(temp_image_list_resized, images[x])
            else:
                new_image_list = composite_masked(cropped_images_scaled[x], images[x])

            image_size = cropped_image_list[x].size()
            image_width = int(image_size[2])
            image_height = int(image_size[1])

            samples = new_image_list.movedim(-1, 1)
            new_image_list_resized = common_upscale(samples, image_width, image_height, "lanczos", "disabled")
            new_image_list_resized = new_image_list_resized.movedim(1,-1)

            if apply_masks:
                new_image_list_resized = composite_masked(cropped_image_list[x], new_image_list_resized, mask=cropped_masks[x])

            image_list_rgba = image_paste_crop_location(original_images[x], new_image_list_resized, top[x], left[x], right[x], bottom[x])

            image_list.append(image_to_rgb(image_list_rgba))


        return (list_to_batch(image_list),)


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
                "destination": (["main_destination", "blank_image"],),
                "resize_sources": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "run"

    CATEGORY = "AIR Nodes"

    def run(self, main_destination, sources, masks, destination, resize_sources):

        if destination == "blank_image":
            size = get_image_size(main_destination)
            width_B = int(size[0])
            height_B = int(size[1])
            composited_image = create_white(width_B,height_B)
        else:
            composited_image = main_destination

        source_list = batch_to_list(sources)
        mask_list = mask_to_list(masks)

        for x in range(len(source_list)):
            composited_image = composite_masked(composited_image, source_list[x], resize_source=resize_sources, mask=mask_list[x])

        return (composited_image,)

NODE_CLASS_MAPPINGS = {
    "TargetLocationCrop": target_location_crop,
    "TargetLocationPaste": target_location_paste,
    "ImageCompositeChained": image_composite_chained,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TargetLocationCrop": "Target Location (Crop)",
    "TargetLocationPaste": "Target Location (Paste)",
    "ImageCompositeChained": "Image Composite Chained",
}
