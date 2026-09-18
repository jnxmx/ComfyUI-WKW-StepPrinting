"""
Wong Kar-Wai / Christopher Doyle Film & Lens Styling Engine.
Includes:
- Halation and neon bloom (photochemical rem-jet halation)
- Signature 90s Hong Kong color grading palettes
- Vintage lens chromatic aberration (radial prism fringe)
- Organic 35mm film grain
"""

import math
import torch
import torch.nn.functional as F


def apply_neon_bloom_and_halation(
    img: torch.Tensor,
    bloom_intensity: float = 0.4,
    halation_tint: tuple = (1.2, 0.4, 0.1),
    threshold: float = 0.65
) -> torch.Tensor:
    """
    Simulates vintage lens neon bloom and photochemical film halation.
    img: [B, C, H, W] in range [0, 1]
    """
    if bloom_intensity <= 0.001:
        return img

    B, C, H, W = img.shape
    # Extract bright highlights above threshold
    lum = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    high = torch.clamp((lum - threshold) / (1.0 - threshold + 1e-4), 0.0, 1.0)
    high_rgb = img * high

    # Multi-scale downsample and blur for wide diffuse bloom
    # Level 1 (sharp halation fringe)
    h1 = F.avg_pool2d(high_rgb, kernel_size=3, stride=1, padding=1)
    # Level 2 (medium neon glow)
    h2 = F.interpolate(high_rgb, scale_factor=0.25, mode="bilinear", align_corners=False)
    h2 = F.avg_pool2d(h2, kernel_size=5, stride=1, padding=2)
    h2 = F.interpolate(h2, size=(H, W), mode="bilinear", align_corners=False)

    # Tint halation warm/orange-red (film rem-jet layer characteristic)
    device = img.device
    tint = torch.tensor(halation_tint, device=device, dtype=img.dtype).view(1, 3, 1, 1)
    halation = (h1 * 0.6 + h2 * 0.8) * tint * bloom_intensity

    # Screen / Additive blend
    out = img + halation
    return torch.clamp(out, 0.0, 1.0)


def apply_chromatic_aberration(img: torch.Tensor, amount: float = 0.3) -> torch.Tensor:
    """
    Simulates vintage prime lens (e.g. Canon K-35) radial chromatic aberration.
    Shifts Red channel outward and Blue channel inward.
    """
    if amount <= 0.001:
        return img

    B, C, H, W = img.shape
    device = img.device

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device, dtype=img.dtype),
        torch.linspace(-1.0, 1.0, W, device=device, dtype=img.dtype),
        indexing="ij"
    )
    base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1) # [B, H, W, 2]

    # Radial distortion factor r^2
    r_sq = (grid_x ** 2 + grid_y ** 2).unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1, 1)

    scale_r = 1.0 + 0.012 * amount * r_sq
    scale_b = 1.0 - 0.012 * amount * r_sq

    grid_r = base_grid * scale_r
    grid_b = base_grid * scale_b

    warped_r = F.grid_sample(img[:, 0:1], grid_r, mode="bilinear", padding_mode="border", align_corners=True)
    warped_g = img[:, 1:2]
    warped_b = F.grid_sample(img[:, 2:3], grid_b, mode="bilinear", padding_mode="border", align_corners=True)

    return torch.cat([warped_r, warped_g, warped_b], dim=1)


def apply_wkw_color_palette(img: torch.Tensor, preset: str = "chungking_green_amber") -> torch.Tensor:
    """
    Applies Christopher Doyle color grading transforms.
    """
    if preset == "none":
        return img

    B, C, H, W = img.shape
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    if preset == "chungking_green_amber":
        # Greenish-cyan in shadows, warm amber in highlights
        shadow_mask = 1.0 - torch.clamp(lum * 1.5, 0.0, 1.0)
        high_mask = torch.clamp((lum - 0.3) * 1.4, 0.0, 1.0)

        r = r + high_mask * 0.18 - shadow_mask * 0.08
        g = g + shadow_mask * 0.14 + high_mask * 0.07
        b = b - high_mask * 0.15 + shadow_mask * 0.05

    elif preset == "fallen_angels_night":
        # Deep cool blues/cyan with punchy fluorescent neon highlights
        shadow_mask = 1.0 - torch.clamp(lum * 1.4, 0.0, 1.0)
        r = r - shadow_mask * 0.12 + 0.05
        g = g + shadow_mask * 0.05
        b = b + shadow_mask * 0.20

    elif preset == "in_the_mood_warm":
        # Rich tungsten warmth, nostalgic golden red tones
        r = r * 1.15 + 0.04
        g = g * 0.98 + 0.01
        b = b * 0.82 - 0.02

    # S-curve contrast
    out = torch.cat([r, g, b], dim=1)
    out = torch.clamp(out, 0.0, 1.0)
    out = out ** 1.15 # slight contrast boost
    return out


def apply_film_grain(img: torch.Tensor, intensity: float = 0.12) -> torch.Tensor:
    """
    Adds dynamic 35mm photochemical film grain.
    """
    if intensity <= 0.001:
        return img

    device = img.device
    noise = torch.randn_like(img, device=device) * intensity
    # Grain is slightly stronger in midtones than deep crushed blacks or clipped highlights
    lum = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    mid_weight = 4.0 * lum * (1.0 - lum) # peak at 0.5

    out = img + noise * mid_weight
    return torch.clamp(out, 0.0, 1.0)
