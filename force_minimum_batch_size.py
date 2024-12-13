from torch import Tensor
import torch

class minimum_batch_size:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "minimum_batch_size" :("INT", {
                    "default": 8,
                    "step":1,
                    "display": "number"
                }),
                "images": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "original_batch_size")

    FUNCTION = "run"

    CATEGORY = "AIR Nodes"

    def run(self, minimum_batch_size, images: Tensor):
        frame_count = len(images)
        batch_difference = minimum_batch_size - frame_count + 1

        if frame_count < minimum_batch_size:
            repeat_amount = 1
            if batch_difference < 0:
                repeat_amount = 1
            else:
                repeat_amount = batch_difference

            first_frames = images[:-1]
            last_frame = images[-1:]
            x_last_frame = last_frame.repeat((repeat_amount, 1, 1, 1))

            forced_batch = torch.cat((first_frames, x_last_frame), dim=0)

            return (forced_batch, frame_count)
        else:
            return (images, frame_count)


NODE_CLASS_MAPPINGS = {"ForceMinimumBatchSize": minimum_batch_size}

NODE_DISPLAY_NAME_MAPPINGS = {"ForceMinimumBatchSize": "Force Minimum Batch Size"}
