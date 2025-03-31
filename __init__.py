from .force_minimum_batch_size import *
from .prompt_nodes import *
from .target_location import *
from .match_image_count_to_mask_count import *
from .paralax_test import *
from .manga_segmentation import *
from .mask_fill import *
from .line_detection import *
from .yolo_detect import *
from .extract_blacks_lines import *
from .displace_image import *
from .brightness_contrast_saturation import *
from .LTXV_add_AIR_guide import *

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
    "MangaPanelSegmentationNode": MangaPanelSegmentationNode,
    "Mask_Fill_Region": Mask_Fill_Region,
    "LineDetection": LineDetection,
    "Yolov8Detection": Yolov8Detection,
    "ExtractBlackLines": extract_lines,
    "DisplaceImageCPU": displace_image,
    "DisplaceImageGPU": torch_displace_image,
    "GPUTargetLocationCrop": tensor_target_location_crop,
    "GPUTargetLocationPaste": tensor_target_location_paste,
    "BrightnessContrastSaturation": BrightnessContrastSaturation,
    "LTXVAddGuideAIR": LTXVAddGuideAIR,

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
    "MangaPanelSegmentationNode": "Manga Panel Segmentation Node",
    "Mask_Fill_Region": "Mask Fill Region",
    "LineDetection": "Line Detection",
    "Yolov8Detection": "Yolov8 Detection",
    "ExtractBlackLines": "Extract Black Lines",
    "DisplaceImageCPU": "Displace Image CPU",
    "DisplaceImageGPU": "Displace Image GPU",
    "GPUTargetLocationCrop": "GPU Target Location Crop",
    "GPUTargetLocationPaste": "GPU Target Location Paste",
    "BrightnessContrastSaturation": "Brightness Contrast Saturation",
    "LTXVAddGuideAIR": "LTXV Add Guide AIR",

}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']