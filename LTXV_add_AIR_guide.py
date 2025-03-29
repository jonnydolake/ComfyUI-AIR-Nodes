import io
import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.model_sampling
import comfy.utils
import math
import numpy as np
import av
from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier, latent_to_pixel_coords


#forked from comfyui git checkout 889519971fe530abbdc689af20aa439c5e99875f

def conditioning_get_any_value(conditioning, key, default=None):
    for t in conditioning:
        if key in t[1]:
            return t[1][key]
    return default


def get_noise_mask(latent):
    noise_mask = latent.get("noise_mask", None)
    latent_image = latent["samples"]
    if noise_mask is None:
        batch_size, _, latent_length, _, _ = latent_image.shape
        noise_mask = torch.ones(
            (batch_size, 1, latent_length, 1, 1),
            dtype=torch.float32,
            device=latent_image.device,
        )
    else:
        noise_mask = noise_mask.clone()
    return noise_mask

def get_keyframe_idxs(cond):
    keyframe_idxs = conditioning_get_any_value(cond, "keyframe_idxs", None)
    if keyframe_idxs is None:
        return None, 0
    num_keyframes = torch.unique(keyframe_idxs[:, 0]).shape[0]
    return keyframe_idxs, num_keyframes

class LTXVAddGuideAIR:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE",),
                             "latent": ("LATENT",),
                             "image": ("IMAGE", {"tooltip": "Image or video to condition the latent video on. Must be 8*n + 1 frames." \
                                                 "If the video is not 8*n + 1 frames, it will be cropped to the nearest 8*n + 1 frames."}),
                             "frame_idx": ("INT", {"default": 0, "min": -9999, "max": 9999,
                                                   "tooltip": "Frame index to start the conditioning at. Must be divisible by 8. " \
                                                   "If a frame is not divisible by 8, it will be rounded down to the nearest multiple of 8. " \
                                                   "Negative values are counted from the end of the video."}),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             }
            }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    CATEGORY = "AIR Nodes"
    FUNCTION = "generate"

    def __init__(self):
        self._num_prefix_frames = 2
        self._patchifier = SymmetricPatchifier(1)

    def encode(self, vae, latent_width, latent_height, images, scale_factors):
        time_scale_factor, width_scale_factor, height_scale_factor = scale_factors
        images = images[:(images.shape[0] - 1) // time_scale_factor * time_scale_factor + 1]
        pixels = comfy.utils.common_upscale(images.movedim(-1, 1), latent_width * width_scale_factor, latent_height * height_scale_factor, "bilinear", crop="disabled").movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)
        return encode_pixels, t

    def get_latent_index(self, cond, latent_length, frame_idx, scale_factors):
        time_scale_factor, _, _ = scale_factors
        _, num_keyframes = get_keyframe_idxs(cond)
        latent_count = latent_length - num_keyframes
        frame_idx = frame_idx if frame_idx >= 0 else max((latent_count - 1) * 8 + 1 + frame_idx, 0)
        frame_idx = frame_idx // time_scale_factor * time_scale_factor # frame index must be divisible by 8

        latent_idx = (frame_idx + time_scale_factor - 1) // time_scale_factor

        return frame_idx, latent_idx

    def add_keyframe_index(self, cond, frame_idx, guiding_latent, scale_factors):
        keyframe_idxs, _ = get_keyframe_idxs(cond)
        _, latent_coords = self._patchifier.patchify(guiding_latent)
        pixel_coords = latent_to_pixel_coords(latent_coords, scale_factors, True)
        pixel_coords[:, 0] += frame_idx
        if keyframe_idxs is None:
            keyframe_idxs = pixel_coords
        else:
            keyframe_idxs = torch.cat([keyframe_idxs, pixel_coords], dim=2)
        return node_helpers.conditioning_set_values(cond, {"keyframe_idxs": keyframe_idxs})

    def append_keyframe(self, positive, negative, frame_idx, latent_image, noise_mask, guiding_latent, strength, scale_factors):
        positive = self.add_keyframe_index(positive, frame_idx, guiding_latent, scale_factors)
        negative = self.add_keyframe_index(negative, frame_idx, guiding_latent, scale_factors)

        mask = torch.full(
            (noise_mask.shape[0], 1, guiding_latent.shape[2], 1, 1),
            1.0 - strength,
            dtype=noise_mask.dtype,
            device=noise_mask.device,
        )

        latent_image = torch.cat([latent_image, guiding_latent], dim=2)
        noise_mask = torch.cat([noise_mask, mask], dim=2)
        return positive, negative, latent_image, noise_mask

    def replace_latent_frames(self, latent_image, noise_mask, guiding_latent, latent_idx, strength):
        cond_length = guiding_latent.shape[2]
        assert latent_image.shape[2] >= latent_idx + cond_length, "Conditioning frames exceed the length of the latent sequence."

        mask = torch.full(
            (noise_mask.shape[0], 1, cond_length, 1, 1),
            1.0 - strength,
            dtype=noise_mask.dtype,
            device=noise_mask.device,
        )

        latent_image = latent_image.clone()
        noise_mask = noise_mask.clone()

        latent_image[:, :, latent_idx : latent_idx + cond_length] = guiding_latent
        noise_mask[:, :, latent_idx : latent_idx + cond_length] = mask

        return latent_image, noise_mask

    def generate(self, positive, negative, vae, latent, image, frame_idx, strength):
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        image, t = self.encode(vae, latent_width, latent_height, image, scale_factors)

        frame_idx, latent_idx = self.get_latent_index(positive, latent_length, frame_idx, scale_factors)
        assert latent_idx + t.shape[2] <= latent_length, "Conditioning frames exceed the length of the latent sequence."

        if frame_idx == 0:
            latent_image, noise_mask = self.replace_latent_frames(latent_image, noise_mask, t, latent_idx, strength)
            return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask},)


        num_prefix_frames = min(self._num_prefix_frames, t.shape[2])

        positive, negative, latent_image, noise_mask = self.append_keyframe(
            positive,
            negative,
            frame_idx,
            latent_image,
            noise_mask,
            t[:, :, :num_prefix_frames],
            strength,
            scale_factors,
        )

        latent_idx += num_prefix_frames

        t = t[:, :, num_prefix_frames:]
        if t.shape[2] == 0:
            return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask},)

        latent_image, noise_mask = self.replace_latent_frames(
            latent_image,
            noise_mask,
            t,
            latent_idx,
            strength,
        )

        return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask},)
    
NODE_CLASS_MAPPINGS = {
    "LTXVAddGuideAIR": LTXVAddGuideAIR,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVAddGuideAIR": "LTXV Add Guide AIR",

}