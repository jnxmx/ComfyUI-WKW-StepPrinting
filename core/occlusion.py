"""
Occlusion & Disocclusion Estimation and Warping Engine.
Implements forward-backward optical flow consistency checks,
bilinear sub-pixel warping, and soft occlusion confidence masks.
"""

import torch
import torch.nn.functional as F


def backward_warp(x: torch.Tensor, flow: torch.Tensor, padding_mode: str = "border") -> torch.Tensor:
    """
    Warps image or feature map x [B, C, H, W] backwards using optical flow [B, 2, H, W].
    flow[:, 0] is horizontal displacement (dx), flow[:, 1] is vertical displacement (dy).
    """
    B, C, H, W = x.shape
    device = x.device

    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, device=device, dtype=x.dtype),
        torch.arange(0, W, device=device, dtype=x.dtype),
        indexing="ij"
    )
    grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).repeat(B, 1, 1, 1) # [B, H, W, 2]

    # Add flow displacement (permute [B, 2, H, W] -> [B, H, W, 2])
    vgrid = grid + flow.permute(0, 2, 3, 1)

    # Normalize to [-1, 1] range for grid_sample
    vgrid_norm = torch.empty_like(vgrid)
    vgrid_norm[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_norm[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0

    warped = F.grid_sample(x, vgrid_norm, mode="bilinear", padding_mode=padding_mode, align_corners=True)
    return warped


def compute_occlusion_mask(flow_fwd: torch.Tensor, flow_bwd: torch.Tensor, threshold: float = 1.5) -> torch.Tensor:
    """
    Computes soft occlusion confidence mask using forward-backward flow consistency.
    Returns:
        mask: [B, 1, H, W] in range [0, 1].
              1.0 means fully visible / unoccluded.
              0.0 means fully occluded.
    """
    # Warp backward flow into forward frame coordinates
    flow_bwd_warped = backward_warp(flow_bwd, flow_fwd)

    # Consistency vector error: u_fwd(x) + u_bwd(x + u_fwd(x))
    flow_diff = flow_fwd + flow_bwd_warped
    error_sq = torch.sum(flow_diff ** 2, dim=1, keepdim=True) # [B, 1, H, W]

    # Relative error penalty: large motions tolerate slightly larger subpixel errors
    flow_mag_sq = torch.sum(flow_fwd ** 2, dim=1, keepdim=True)
    scale = (threshold ** 2) + 0.02 * flow_mag_sq

    # Soft exponential confidence mask
    mask = torch.exp(-0.5 * error_sq / torch.clamp(scale, min=1e-4))
    return mask


def compute_disocclusion_mask(flow_fwd: torch.Tensor, flow_bwd: torch.Tensor, threshold: float = 1.5) -> torch.Tensor:
    """
    Computes disocclusion mask (areas revealed in frame 2 that were occluded in frame 1).
    Returns:
        mask: [B, 1, H, W] in range [0, 1] (1.0 = disoccluded).
    """
    flow_fwd_warped = backward_warp(flow_fwd, flow_bwd)
    flow_diff = flow_bwd + flow_fwd_warped
    error_sq = torch.sum(flow_diff ** 2, dim=1, keepdim=True)

    flow_mag_sq = torch.sum(flow_bwd ** 2, dim=1, keepdim=True)
    scale = (threshold ** 2) + 0.02 * flow_mag_sq

    disocc = 1.0 - torch.exp(-0.5 * error_sq / torch.clamp(scale, min=1e-4))
    return disocc
