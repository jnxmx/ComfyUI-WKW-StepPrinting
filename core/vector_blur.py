"""
Vector Path & Continuous Shutter Integration Engine.
Computes occlusion-aware sub-pixel optical flow motion blur along vector trajectories,
with customizable shutter curves and highlight streak preservation.
"""

import math
import torch
import torch.nn.functional as F

try:
    from .occlusion import backward_warp, compute_occlusion_mask
except (ImportError, ValueError):
    from core.occlusion import backward_warp, compute_occlusion_mask


def generate_shutter_weights(num_samples: int, curve_type: str = "trailing_decay", decay_rate: float = 2.5) -> torch.Tensor:
    """
    Generates normalized 1D shutter exposure weights along the temporal integration path.
    """
    if num_samples <= 1:
        return torch.tensor([1.0], dtype=torch.float32)

    t = torch.linspace(0.0, 1.0, steps=num_samples)

    if curve_type == "trailing_decay":
        # Exponential decay backwards: frame 2 is current, trailing streak extends backwards
        weights = torch.exp(-decay_rate * (1.0 - t))
    elif curve_type == "gaussian":
        sigma = 0.25
        weights = torch.exp(-0.5 * ((t - 0.5) / sigma) ** 2)
    elif curve_type == "triangle":
        weights = 1.0 - 2.0 * torch.abs(t - 0.5)
        weights = torch.clamp(weights, min=0.02)
    else:
        # Uniform box shutter
        weights = torch.ones_like(t)

    weights = weights / torch.sum(weights)
    return weights


def integrate_vector_path(
    img1: torch.Tensor,
    img2: torch.Tensor,
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
    shutter_fraction: float = 1.0,
    num_samples: int = 12,
    shutter_curve: str = "trailing_decay",
    occlusion_aware: bool = True,
    occlusion_threshold: float = 1.5,
    highlight_preservation: float = 0.3
) -> torch.Tensor:
    """
    Integrates sub-pixel vector trajectories between img1 and img2.
    img1, img2: [B, C, H, W] in range [0, 1]
    flow_fwd: [B, 2, H, W] (img1 -> img2)
    flow_bwd: [B, 2, H, W] (img2 -> img1)
    shutter_fraction: fraction of inter-frame motion to smear across
    num_samples: sub-step count along trajectory
    Returns:
        blurred: [B, C, H, W] in range [0, 1]
    """
    B, C, H, W = img1.shape
    device = img1.device

    num_samples = max(2, int(num_samples))
    weights = generate_shutter_weights(num_samples, curve_type=shutter_curve).to(device)

    # Compute soft occlusion masks if enabled
    if occlusion_aware:
        mask_occ1 = compute_occlusion_mask(flow_fwd, flow_bwd, threshold=occlusion_threshold, min_weight=0.1)
        mask_occ2 = compute_occlusion_mask(flow_bwd, flow_fwd, threshold=occlusion_threshold, min_weight=0.1)
    else:
        mask_occ1 = torch.ones((B, 1, H, W), device=device, dtype=img1.dtype)
        mask_occ2 = torch.ones((B, 1, H, W), device=device, dtype=img2.dtype)

    accum_img = torch.zeros_like(img1)
    accum_weight = torch.zeros((B, 1, H, W), device=device, dtype=img1.dtype)

    # Scale flow by shutter exposure fraction
    scaled_fwd = flow_fwd * shutter_fraction
    scaled_bwd = flow_bwd * shutter_fraction

    for i in range(num_samples):
        # Normalized time coordinate s from 0.0 (frame 1) to 1.0 (frame 2)
        s = float(i) / float(num_samples - 1)
        w_curve = weights[i]

        # Bidirectional sub-pixel flow warps
        flow_to_1 = -s * scaled_fwd
        warp1 = backward_warp(img1, flow_to_1)

        flow_to_2 = -(1.0 - s) * scaled_bwd
        warp2 = backward_warp(img2, flow_to_2)

        # Occlusion-weighted directional blending
        m1 = backward_warp(mask_occ1, flow_to_1)
        m2 = backward_warp(mask_occ2, flow_to_2)

        # Ensure minimum weight floor so division is always safe and never collapses to black
        w1 = (1.0 - s) * torch.clamp(m1, min=0.08)
        w2 = s * torch.clamp(m2, min=0.08)

        denom = w1 + w2
        interp = (w1 * warp1 + w2 * warp2) / denom

        # Highlight preservation (boost neon / specular highlights along the trail)
        if highlight_preservation > 0.0:
            lum = 0.299 * interp[:, 0:1] + 0.587 * interp[:, 1:2] + 0.114 * interp[:, 2:3]
            sample_weight = w_curve * (1.0 + highlight_preservation * torch.clamp(lum, 0.0, 1.0) ** 2)
        else:
            sample_weight = w_curve

        accum_img += interp * sample_weight
        accum_weight += sample_weight

    blurred = accum_img / torch.clamp(accum_weight, min=1e-5)
    return torch.clamp(blurred, 0.0, 1.0)
