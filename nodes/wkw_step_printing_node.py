"""
ComfyUI Node: Wong Kar-Wai Step Printing (Optical Flow).
"""

import torch
try:
    from ..core.step_printer import StepPrinter
except (ImportError, ValueError):
    from core.step_printer import StepPrinter


class WKWStepPrintingNode:
    """
    Converts any video sequence into Wong Kar-Wai style step-printed slow motion.
    Computes deep optical flow, forward-backward occlusion consistency, continuous vector paths,
    and shutter angle aperture motion smearing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "input_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.5}),
                "target_capture_fps": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 60.0, "step": 0.5}),
                "shutter_angle": ("FLOAT", {"default": 360.0, "min": 0.0, "max": 720.0, "step": 5.0, "tooltip": "180=standard cinema, 360=full open shutter, 720=extreme double-frame light streak"}),
                "optical_flow_method": ([
                    "RAFT-Small (Deep Learning)",
                    "RAFT-Large (Deep Learning)",
                    "DIS (Fast OpenCV)",
                    "Farneback (OpenCV)"
                ], {"default": "RAFT-Small (Deep Learning)"}),
                "flow_samples": ("INT", {"default": 12, "min": 2, "max": 48, "step": 1, "tooltip": "Sub-pixel integration steps along vector trajectories"}),
                "occlusion_aware": ("BOOLEAN", {"default": True, "tooltip": "Forward-backward consistency check to prevent edge bleed and ghosting"}),
                "occlusion_threshold": ("FLOAT", {"default": 1.5, "min": 0.2, "max": 8.0, "step": 0.1}),
                "shutter_curve": ([
                    "trailing_decay",
                    "gaussian",
                    "box",
                    "triangle"
                ], {"default": "trailing_decay", "tooltip": "trailing_decay = iconic glowing neon streak fading behind moving objects"}),
                "cadence_mode": ([
                    "step_print_repeat",
                    "decimate_to_target_fps",
                    "slow_motion_stretch"
                ], {"default": "step_print_repeat", "tooltip": "step_print_repeat = standard timeline with repeated held frames (AAAA BBBB); decimate = true low fps; slow_motion_stretch = extended slow motion"}),
                "step_repeat_count": ("INT", {"default": 4, "min": 1, "max": 24, "step": 1, "tooltip": "Hold duration per capture frame when in slow_motion_stretch mode"}),
                "vector_blur_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"})
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "MASK")
    RETURN_NAMES = ("images", "output_fps", "occlusion_masks")
    FUNCTION = "apply_step_printing"
    CATEGORY = "Wong Kar-Wai VFX"

    def apply_step_printing(
        self,
        images: torch.Tensor,
        input_fps: float,
        target_capture_fps: float,
        shutter_angle: float,
        optical_flow_method: str,
        flow_samples: int,
        occlusion_aware: bool,
        occlusion_threshold: float,
        shutter_curve: str,
        cadence_mode: str,
        step_repeat_count: int,
        vector_blur_intensity: float,
        device: str
    ):
        # ComfyUI image format: [B, H, W, C] in range [0, 1]
        # Permute to PyTorch standard: [B, C, H, W]
        orig_device = images.device
        x = images.permute(0, 3, 1, 2).contiguous()

        out_frames, out_fps, out_masks = StepPrinter.process(
            frames=x,
            input_fps=input_fps,
            target_capture_fps=target_capture_fps,
            shutter_angle=shutter_angle,
            optical_flow_method=optical_flow_method,
            flow_samples=flow_samples,
            occlusion_aware=occlusion_aware,
            occlusion_threshold=occlusion_threshold,
            vector_blur_intensity=vector_blur_intensity,
            shutter_curve=shutter_curve,
            cadence_mode=cadence_mode,
            step_repeat_count=step_repeat_count,
            device=device
        )

        # Permute back to [B, H, W, C]
        res_images = out_frames.permute(0, 2, 3, 1).to(orig_device)
        # ComfyUI mask format: [B, H, W]
        res_masks = out_masks.squeeze(1).to(orig_device)

        return (res_images, float(out_fps), res_masks)
