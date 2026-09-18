"""
ComfyUI Node: Wong Kar-Wai Film & Lens Style.
"""

import torch
try:
    from ..core.color_grade import (
        apply_neon_bloom_and_halation,
        apply_chromatic_aberration,
        apply_wkw_color_palette,
        apply_film_grain
    )
except (ImportError, ValueError):
    from core.color_grade import (
        apply_neon_bloom_and_halation,
        apply_chromatic_aberration,
        apply_wkw_color_palette,
        apply_film_grain
    )


class WKWGradingNode:
    """
    Simulates Christopher Doyle / Wong Kar-Wai cinematography aesthetics:
    vintage prime lens halation & neon bloom, 90s Hong Kong film grading palettes,
    radial chromatic aberration, and photochemical 35mm grain.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "color_preset": ([
                    "chungking_green_amber",
                    "fallen_angels_night",
                    "in_the_mood_warm",
                    "none"
                ], {"default": "chungking_green_amber"}),
                "neon_bloom_intensity": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05}),
                "bloom_threshold": ("FLOAT", {"default": 0.65, "min": 0.2, "max": 0.95, "step": 0.02}),
                "chromatic_aberration": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05}),
                "film_grain": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.8, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.15, "min": 0.5, "max": 2.0, "step": 0.05}),
                "saturation": ("FLOAT", {"default": 1.10, "min": 0.0, "max": 2.5, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply_style"
    CATEGORY = "Wong Kar-Wai VFX"

    def apply_style(
        self,
        images: torch.Tensor,
        color_preset: str,
        neon_bloom_intensity: float,
        bloom_threshold: float,
        chromatic_aberration: float,
        film_grain: float,
        contrast: float,
        saturation: float
    ):
        orig_device = images.device
        # [B, H, W, C] -> [B, C, H, W]
        x = images.permute(0, 3, 1, 2).contiguous()

        # 1. Color Palette Grade
        x = apply_wkw_color_palette(x, preset=color_preset)

        # 2. Contrast adjustment
        if abs(contrast - 1.0) > 0.01:
            mean = 0.5
            x = (x - mean) * contrast + mean
            x = torch.clamp(x, 0.0, 1.0)

        # 3. Saturation adjustment
        if abs(saturation - 1.0) > 0.01:
            lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            x = lum + (x - lum) * saturation
            x = torch.clamp(x, 0.0, 1.0)

        # 4. Neon Bloom & Rem-jet Halation
        x = apply_neon_bloom_and_halation(
            x,
            bloom_intensity=neon_bloom_intensity,
            threshold=bloom_threshold
        )

        # 5. Vintage Lens Radial Chromatic Aberration
        x = apply_chromatic_aberration(x, amount=chromatic_aberration)

        # 6. Photochemical Film Grain
        x = apply_film_grain(x, intensity=film_grain)

        # Permute back to [B, H, W, C]
        res = x.permute(0, 2, 3, 1).to(orig_device)
        return (res,)
