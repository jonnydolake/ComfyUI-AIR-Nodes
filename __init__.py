from .force_minimum_batch_size import *
from .prompt_nodes import *
from .target_location import *
from .match_image_count_to_mask_count import *
from .paralax_test import *
from .paralax_gpu_test import *

NODE_CLASS_MAPPINGS = {
    "string_list_to_prompt_schedule": string_list_to_prompt_schedule,
    "ForceMinimumBatchSize": minimum_batch_size,
    "TargetLocationCrop": target_location_crop,
    "TargetLocationPaste": target_location_paste,
    "ImageCompositeChained": image_composite_chained,
    "MatchImageCountToMaskCount": match_image_count_to_mask_count,
    "RandomCharacterPrompts": random_character_prompts,
    "ParallaxTest": parallax_test,
    "easy_parallax": easy_parallax,
    "ParallaxGPUTest": ParallaxGPUTest,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "string_list_to_prompt_schedule": "String List To Prompt Schedule",
    "ForceMinimumBatchSize": "Force Minimum Batch Size",
    "TargetLocationCrop": "Target Location (Crop)",
    "TargetLocationPaste": "Target Location (Paste)",
    "ImageCompositeChained": "Image Composite Chained",
    "MatchImageCountToMaskCount": "Match Image Count To Mask Count",
    "RandomCharacterPrompts": "Random Character Prompts",
    "ParallaxTest": "Parallax Test",
    "easy_parallax": "Easy Parallax",
    "ParallaxGPUTest": "Parallax GPU Test",


}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']