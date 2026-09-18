"""
Optical Flow Engine for ComfyUI-WKW-StepPrinting.
Supports PyTorch Deep Learning (torchvision RAFT-Small, RAFT-Large) and OpenCV DIS/Farneback fallback.
Optimized for NVIDIA GPUs (CUDA) with support for MPS and CPU.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torchvision.models.optical_flow as of
    HAS_TORCHVISION_OF = True
except ImportError:
    HAS_TORCHVISION_OF = False


class FlowEngine:
    """Manages model loading, caching, inference, and fallbacks for optical flow."""
    _models = {}

    @classmethod
    def get_device(cls, requested="auto"):
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        elif requested == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif requested == "mps":
            return torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        return torch.device("cpu")

    @classmethod
    def get_raft_model(cls, model_type="raft_small", device=None):
        if device is None:
            device = cls.get_device("auto")
        key = (model_type, str(device))
        if key in cls._models:
            return cls._models[key]

        if not HAS_TORCHVISION_OF:
            raise RuntimeError("torchvision.models.optical_flow is not available. Please install/upgrade torchvision.")

        if model_type == "raft_large":
            weights = of.Raft_Large_Weights.DEFAULT
            model = of.raft_large(weights=weights)
        else:
            weights = of.Raft_Small_Weights.DEFAULT
            model = of.raft_small(weights=weights)

        model = model.to(device).eval()
        cls._models[key] = (model, weights.transforms(), device)
        return cls._models[key]

    @classmethod
    def compute_raft_flow(cls, img1, img2, model_type="raft_small", device=None):
        """
        Compute optical flow from img1 to img2 using RAFT.
        img1, img2: [B, C, H, W] in range [0, 1]
        Returns: flow [B, 2, H, W] in pixels.
        """
        model, transforms, model_device = cls.get_raft_model(model_type, device)
        orig_device = img1.device

        B, C, H, W = img1.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8

        t_img1 = img1.to(model_device)
        t_img2 = img2.to(model_device)
        if t_img1.max() <= 1.05:
            t_img1 = t_img1 * 255.0
            t_img2 = t_img2 * 255.0

        if pad_h > 0 or pad_w > 0:
            t_img1 = F.pad(t_img1, (0, pad_w, 0, pad_h), mode="replicate")
            t_img2 = F.pad(t_img2, (0, pad_w, 0, pad_h), mode="replicate")

        t_img1, t_img2 = transforms(t_img1, t_img2)

        with torch.inference_mode():
            flow_predictions = model(t_img1, t_img2)
            flow = flow_predictions[-1]

        if pad_h > 0 or pad_w > 0:
            flow = flow[:, :, :H, :W]

        return flow.to(orig_device)

    @classmethod
    def compute_opencv_flow(cls, img1, img2, method="dis"):
        """
        Fallback optical flow using OpenCV DIS or Farneback.
        img1, img2: [B, C, H, W] torch.Tensor in [0, 1]
        Returns: flow [B, 2, H, W] in pixels.
        """
        if not HAS_CV2:
            raise RuntimeError("OpenCV (cv2) is required for DIS/Farneback optical flow.")

        B, C, H, W = img1.shape
        flows = []

        for b in range(B):
            f1 = (img1[b].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            f2 = (img2[b].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
            g2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)

            if method == "dis":
                dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                flow_np = dis.calc(g1, g2, None)
            else:
                flow_np = cv2.calcOpticalFlowFarneback(
                    g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
            flows.append(torch.from_numpy(flow_np).permute(2, 0, 1))

        flow_tensor = torch.stack(flows, dim=0).to(img1.device, dtype=torch.float32)
        return flow_tensor

    @classmethod
    def compute_bidirectional_flow(cls, img1, img2, method="RAFT-Small (Deep Learning)", device=None):
        """
        Computes forward (img1 -> img2) and backward (img2 -> img1) optical flow.
        Returns:
            flow_fwd: [B, 2, H, W]
            flow_bwd: [B, 2, H, W]
        """
        if "RAFT-Large" in method:
            flow_fwd = cls.compute_raft_flow(img1, img2, model_type="raft_large", device=device)
            flow_bwd = cls.compute_raft_flow(img2, img1, model_type="raft_large", device=device)
        elif "RAFT-Small" in method or "RAFT" in method:
            flow_fwd = cls.compute_raft_flow(img1, img2, model_type="raft_small", device=device)
            flow_bwd = cls.compute_raft_flow(img2, img1, model_type="raft_small", device=device)
        elif "Farneback" in method:
            flow_fwd = cls.compute_opencv_flow(img1, img2, method="farneback")
            flow_bwd = cls.compute_opencv_flow(img2, img1, method="farneback")
        else:
            flow_fwd = cls.compute_opencv_flow(img1, img2, method="dis")
            flow_bwd = cls.compute_opencv_flow(img2, img1, method="dis")

        return flow_fwd, flow_bwd
