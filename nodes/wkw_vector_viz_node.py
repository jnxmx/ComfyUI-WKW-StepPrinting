"""
ComfyUI Node: Wong Kar-Wai Motion Vector Visualizer & Diagnostics.
"""

import math
import torch
import torch.nn.functional as F
try:
    from ..core.flow_engine import FlowEngine
    from ..core.occlusion import compute_occlusion_mask, compute_disocclusion_mask
except (ImportError, ValueError):
    from core.flow_engine import FlowEngine
    from core.occlusion import compute_occlusion_mask, compute_disocclusion_mask


def flow_to_rgb(flow: torch.Tensor, max_flow: float = None) -> torch.Tensor:
    """
    Converts optical flow [B, 2, H, W] to color-coded HSV visualization [B, 3, H, W] in [0, 1].
    """
    B, _, H, W = flow.shape
    u = flow[:, 0]
    v = flow[:, 1]

    mag = torch.sqrt(u ** 2 + v ** 2)
    angle = torch.atan2(v, u) # range [-pi, pi]

    # Normalize angle to [0, 1] for Hue
    hue = (angle + math.pi) / (2.0 * math.pi)

    if max_flow is None:
        max_flow = torch.quantile(mag.view(B, -1), 0.98, dim=1).view(B, 1, 1)
        max_flow = torch.clamp(max_flow, min=1.0)

    val = torch.clamp(mag / max_flow, 0.0, 1.0)
    sat = torch.ones_like(hue)

    # HSV to RGB conversion
    h6 = hue * 6.0
    c = val * sat
    x = c * (1.0 - torch.abs((h6 % 2.0) - 1.0))
    m = val - c

    r = torch.zeros_like(h6)
    g = torch.zeros_like(h6)
    b = torch.zeros_like(h6)

    case0 = (h6 >= 0.0) & (h6 < 1.0)
    case1 = (h6 >= 1.0) & (h6 < 2.0)
    case2 = (h6 >= 2.0) & (h6 < 3.0)
    case3 = (h6 >= 3.0) & (h6 < 4.0)
    case4 = (h6 >= 4.0) & (h6 < 5.0)
    case5 = (h6 >= 5.0) & (h6 <= 6.0)

    r[case0] = c[case0]; g[case0] = x[case0]; b[case0] = 0.0
    r[case1] = x[case1]; g[case1] = c[case1]; b[case1] = 0.0
    r[case2] = 0.0;      g[case2] = c[case2]; b[case2] = x[case2]
    r[case3] = 0.0;      g[case3] = x[case3]; b[case3] = c[case3]
    r[case4] = x[case4]; g[case4] = 0.0;      b[case4] = c[case4]
    r[case5] = c[case5]; g[case5] = 0.0;      b[case5] = x[case5]

    rgb = torch.stack([r + m, g + m, b + m], dim=1)
    return torch.clamp(rgb, 0.0, 1.0)


class WKWVectorVisualizerNode:
    """
    VFX Diagnostic Node: Inspects optical flow motion vectors and occlusion masks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "optical_flow_method": ([
                    "RAFT-Small (Deep Learning)",
                    "RAFT-Large (Deep Learning)",
                    "DIS (Fast OpenCV)",
                    "Farneback (OpenCV)"
                ], {"default": "RAFT-Small (Deep Learning)"}),
                "occlusion_threshold": ("FLOAT", {"default": 1.5, "min": 0.2, "max": 8.0, "step": 0.1}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"})
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("flow_visualization", "occlusion_mask", "disocclusion_mask")
    FUNCTION = "visualize"
    CATEGORY = "Wong Kar-Wai VFX"

    def visualize(
        self,
        images: torch.Tensor,
        optical_flow_method: str,
        occlusion_threshold: float,
        device: str
    ):
        orig_device = images.device
        x = images.permute(0, 3, 1, 2).contiguous()
        T, C, H, W = x.shape
        target_device = FlowEngine.get_device(device)
        x = x.to(target_device)

        flow_viz_list = []
        occ_list = []
        disocc_list = []

        for k in range(max(1, T - 1)):
            idx_next = min(k + 1, T - 1)
            f1 = x[k:k+1]
            f2 = x[idx_next:idx_next+1]

            if k == idx_next:
                flow_fwd = torch.zeros((1, 2, H, W), device=target_device)
                flow_bwd = torch.zeros((1, 2, H, W), device=target_device)
            else:
                flow_fwd, flow_bwd = FlowEngine.compute_bidirectional_flow(
                    f1, f2, method=optical_flow_method, device=target_device
                )

            rgb_flow = flow_to_rgb(flow_fwd)
            mask_occ = compute_occlusion_mask(flow_fwd, flow_bwd, threshold=occlusion_threshold)
            mask_disocc = compute_disocclusion_mask(flow_fwd, flow_bwd, threshold=occlusion_threshold)

            flow_viz_list.append(rgb_flow)
            occ_list.append(mask_occ.repeat(1, 3, 1, 1))
            disocc_list.append(mask_disocc.repeat(1, 3, 1, 1))

        if T > 1:
            # duplicate last frame to keep length equal to input T
            flow_viz_list.append(flow_viz_list[-1])
            occ_list.append(occ_list[-1])
            disocc_list.append(disocc_list[-1])

        flow_out = torch.cat(flow_viz_list, dim=0).permute(0, 2, 3, 1).to(orig_device)
        occ_out = torch.cat(occ_list, dim=0).permute(0, 2, 3, 1).to(orig_device)
        disocc_out = torch.cat(disocc_list, dim=0).permute(0, 2, 3, 1).to(orig_device)

        return (flow_out, occ_out, disocc_out)
