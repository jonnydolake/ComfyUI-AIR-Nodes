
class match_image_count_to_mask_count:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image" :("IMAGE",),
                "masks": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE","MASK")
    RETURN_NAMES = ("images","masks")

    FUNCTION = "run"

    CATEGORY = "AIR Nodes"

    def run(self, image, masks):
        amount = len(masks)
        s = image.repeat((amount, 1, 1, 1))
        return (s, masks)


NODE_CLASS_MAPPINGS = {"MatchImageCountToMaskCount": match_image_count_to_mask_count}

NODE_DISPLAY_NAME_MAPPINGS = {"MatchImageCountToMaskCount": "Match Image Count To Mask Count"}
