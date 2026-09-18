"""
Vector Path & Continuous Shutter Integration Engine.
Computes true continuous liquid motion blur along vector trajectories using
forward splatting, occlusion-aware Z-ordering, and customizable shutter curves.
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
        sigma = 0.28
        weights = torch.exp(-0.5 * ((t - 0.5) / sigma) ** 2)
    elif curve_type == "triangle":
        weights = 1.0 - 2.0 * torch.abs(t - 0.5)
        weights = torch.clamp(weights, min=0.05)
    else:
        # Uniform box shutter
        weights = torch.ones_like(t)

    weights = weights / torch.sum(weights)
    return weights


def forward_splat_substep(
    img1: torch.Tensor,
    img2: torch.Tensor,
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
    s: float,
    occlusion_aware: bool = True,
    occlusion_threshold: float = 1.5
) -> torch.Tensor:
    """
    Forward-splats pixels from img1 and img2 to intermediate time s in [0, 1].
    Produces a continuous, liquid, physically transported frame without discrete ghosting.
    """
    B, C, H, W = img1.shape
    device = img1.device

    # Grid coordinates
    gy, gx = torch.meshgrid(
        torch.arange(0, H, device=device, dtype=torch.float32),
        torch.arange(0, W, device=device, dtype=torch.float32),
        indexing="ij"
    )

    # Motion magnitude
    mag_fwd = torch.sqrt(flow_fwd[:, 0:1] ** 2 + flow_fwd[:, 1:2] ** 2)
    mag_bwd = torch.sqrt(flow_bwd[:, 0:1] ** 2 + flow_bwd[:, 1:2] ** 2)

    # Occlusion-aware priority weighting (Z-order / Softmax splatting)
    if occlusion_aware:
        # Compute forward-backward consistency
        occ1 = compute_occlusion_mask(flow_fwd, flow_bwd, threshold=occlusion_threshold, min_weight=0.1)
        occ2 = compute_occlusion_mask(flow_bwd, flow_fwd, threshold=occlusion_threshold, min_weight=0.1)
        # Moving foreground with high consistency gets higher Z-depth priority
        imp1 = (1.0 - s) * occ1 * (1.0 + 2.5 * torch.clamp(mag_fwd / 6.0, 0.0, 1.0))
        imp2 = s * occ2 * (1.0 + 2.5 * torch.clamp(mag_bwd / 6.0, 0.0, 1.0))
    else:
        # Equal double-exposure weighting (translucent optical printing)
        imp1 = torch.full((B, 1, H, W), (1.0 - s), device=device, dtype=img1.dtype)
        imp2 = torch.full((B, 1, H, W), s, device=device, dtype=img2.dtype)

    # Target positions from img1 (moving forward by +s * flow_fwd)
    tx1 = (gx.unsqueeze(0) + s * flow_fwd[:, 0]).clamp(0.0, float(W - 1))
    ty1 = (gy.unsqueeze(0) + s * flow_fwd[:, 1]).clamp(0.0, float(H - 1))

    # Target positions from img2 (moving backward by +(1-s) * flow_bwd)
    tx2 = (gx.unsqueeze(0) + (1.0 - s) * flow_bwd[:, 0]).clamp(0.0, float(W - 1))
    ty2 = (gy.unsqueeze(0) + (1.0 - s) * flow_bwd[:, 1]).clamp(0.0, float(H - 1))

    out_img = torch.zeros((B, C, H * W), device=device, dtype=img1.dtype)
    out_w = torch.zeros((B, 1, H * W), device=device, dtype=img1.dtype)

    # Splat both frames onto the intermediate buffer
    for img_src, tx, ty, imp in [(img1, tx1, ty1, imp1), (img2, tx2, ty2, imp2)]:
        x0 = tx.floor().long()
        x1 = (x0 + 1).clamp(0, W - 1)
        y0 = ty.floor().long()
        y1 = (y0 + 1).clamp(0, H - 1)

        wx1 = tx - x0.float()
        wx0 = 1.0 - wx1
        wy1 = ty - y0.float()
        wy0 = 1.0 - wy1

        corners = [
            (y0 * W + x0, wx0 * wy0),
            (y1 * W + x0, wx0 * wy1),
            (y0 * W + x1, wx1 * wy0),
            (y1 * W + x1, wx1 * wy1),
        ]

        for b in range(B):
            imp_b = imp[b, 0].view(-1)
            for idx_c, w_c in corners:
                idx_flat = idx_c[b].view(-1)
                w_eff = (w_c[b].view(-1) * imp_b)

                out_w[b, 0].scatter_add_(0, idx_flat, w_eff)
                for c in range(C):
                    out_img[b, c].scatter_add_(0, idx_flat, img_src[b, c].view(-1) * w_eff)

    out_img = out_img.view(B, C, H, W)
    out_w = out_w.view(B, 1, H, W)

    # Seamless background base for any unfilled gaps / disoccluded voids
    base_bg = (1.0 - s) * img1 + s * img2

    # Smooth normalization
    splat_valid = torch.clamp(out_w, 0.0, 1.0)
    norm_splat = out_img / torch.clamp(out_w, min=1e-4)

    # Composite: where splat landed, use liquid splat; elsewhere smoothly blend base background
    final_frame = splat_valid * norm_splat + (1.0 - splat_valid) * base_bg
    return torch.clamp(final_frame, 0.0, 1.0)


def integrate_vector_path(
    img1: torch.Tensor,
    img2: torch.Tensor,
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
    shutter_fraction: float = 1.0,
    num_samples: int = 16,
    shutter_curve: str = "trailing_decay",
    occlusion_aware: bool = True,
    occlusion_threshold: float = 1.5,
    highlight_preservation: float = 0.35
) -> torch.Tensor:
    """
    Integrates sub-pixel vector trajectories between img1 and img2 across the shutter angle.
    num_samples: sub-step count along trajectory (default: 16)
    Returns:
        blurred: [B, C, H, W] continuous liquid motion blur.
    """
    B, C, H, W = img1.shape
    device = img1.device

    num_samples = max(2, int(num_samples))
    weights = generate_shutter_weights(num_samples, curve_type=shutter_curve).to(device)

    accum_img = torch.zeros_like(img1)
    accum_weight = 0.0

    # Scale flow by active shutter fraction
    scaled_fwd = flow_fwd * shutter_fraction
    scaled_bwd = flow_bwd * shutter_fraction

    for i in range(num_samples):
        # Normalized time coordinate s from 0.0 (frame 1) to 1.0 (frame 2)
        s = float(i) / float(num_samples - 1)
        w_curve = float(weights[i].item())

        # Forward splat continuous sub-step
        sub_frame = forward_splat_substep(
            img1=img1,
            img2=img2,
            flow_fwd=scaled_fwd,
            flow_bwd=scaled_bwd,
            s=s,
            occlusion_aware=occlusion_aware,
            occlusion_threshold=occlusion_threshold
        )

        # Highlight preservation (neon signs / lights trail with intense glow)
        if highlight_preservation > 0.0:
            lum = 0.299 * sub_frame[:, 0:1] + 0.587 * sub_frame[:, 1:2] + 0.114 * sub_frame[:, 2:3]
            boost = 1.0 + highlight_preservation * torch.clamp(lum, 0.0, 1.0) ** 2
            accum_img += sub_frame * (w_curve * boost)
            accum_weight += w_curve * boost.mean().item()
        else:
            accum_img += sub_frame * w_curve
            accum_weight += w_curve

    blurred = accum_img / max(1e-5, accum_weight)
    return torch.clamp(blurred, 0.0, 1.0)
