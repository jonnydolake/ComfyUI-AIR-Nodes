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


class JoinImageLists:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image1" : ("IMAGE", {"forceInput": True}),
                         "image2": ("IMAGE", {"forceInput": True}),
                         },
        }

    RETURN_TYPES = ("IMAGE",)
    INPUT_IS_LIST = (True, True)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "join_lists"

    CATEGORY = "AIR Nodes"

    def join_lists(self, image1, image2):
        values = image1 + image2

        return (values,)


class BatchListToFlatList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image" : ("IMAGE",),
                         #"list2": ("IMAGE", {"forceInput": True}),
                         },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "execute"

    CATEGORY = "AIR Nodes"

    def execute(self, image):
        flat_list = []
        sizes = []
        for img in image:
            sizes.append(img.shape[0])
            flat_list.extend([img[i].unsqueeze(0) for i in range(img.shape[0])])
        return (flat_list, sizes)


class FlatListToBatchList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image" : ("IMAGE",),
                         "sizes": ("INT", {"forceInput": True}),
                         },
        }

    RETURN_TYPES = ("IMAGE",)
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"

    CATEGORY = "AIR Nodes"

    def execute(self, image, sizes):
        batches = []
        idx = 0
        for size in sizes:
            batches.append(torch.cat(image[idx:idx + size], dim=0))  # (B_i, C, H, W)
            idx += size
        return (batches,)


class GetImageFromList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image_list": ("IMAGE", {"forceInput": True}),
                         "index": ("INT", {"default": 0, "min":-1024, "max": 1024, "step": 1}),
                         },
        }

    RETURN_TYPES = ("IMAGE",)
    INPUT_IS_LIST = (True, False)
    #OUTPUT_IS_LIST = (False,)
    FUNCTION = "get_index"

    CATEGORY = "AIR Nodes"

    def get_index(self, image_list, index):
        return (image_list[index[0]],)

    class GetImageFromList:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {"image_list": ("IMAGE", {"forceInput": True}),
                             "index": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
                             },
            }

        RETURN_TYPES = ("IMAGE",)
        INPUT_IS_LIST = (True,)
        # OUTPUT_IS_LIST = (False,)
        FUNCTION = "get_index"

        CATEGORY = "AIR Nodes"

        def get_index(self, image_list, index):
            return (image_list[index[0]],)


class RemoveElementFromList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image_list": ("IMAGE", {"forceInput": True}),
                         "index": ("INT", {"default": 0, "min": -1024, "max": 1024}),
                         },
        }

    RETURN_TYPES = ("IMAGE","IMAGE")
    RETURN_NAMES = ("image", "new_list")
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "get_index"

    CATEGORY = "AIR Nodes"

    def get_index(self, image_list, index):
        new_list = [image_list[x] for x in range(len(image_list)) if x != index[0]]
        return (image_list[index[0]], new_list)


NODE_CLASS_MAPPINGS = {
    "BrightnessContrastSaturation": BrightnessContrastSaturation,
    "JoinImageLists": JoinImageLists,
    "GetImageFromList": GetImageFromList,
    "RemoveElementFromList": RemoveElementFromList,
    "BatchListToFlatList": BatchListToFlatList,
    "FlatListToBatchList": FlatListToBatchList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BrightnessContrastSaturation": "Brightness Contrast Saturation",
    "JoinImageLists": "Join Image Lists",
    "GetImageFromList": "Get Image From List",
    "RemoveElementFromList": "Remove Element From List",
    "BatchListToFlatList": "Batch List To Flat List",
    "FlatListToBatchList": "Flat List To Batch List",
}