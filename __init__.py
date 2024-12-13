from .force_minimum_batch_size import *
from .string_list_to_prompt_schedule import *
from .target_location import *

NODE_CLASS_MAPPINGS = {
    "string_list_to_prompt_schedule": string_list_to_prompt_schedule,
    "ForceMinimumBatchSize": minimum_batch_size,
    "TargetLocationCrop": target_location_crop,
    "TargetLocationPaste": target_location_paste,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "string_list_to_prompt_schedule": "String List To Prompt Schedule",
    "ForceMinimumBatchSize": "Force Minimum Batch Size",
    "TargetLocationCrop": "Target Location (Crop)",
    "TargetLocationPaste": "Target Location (Paste)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']