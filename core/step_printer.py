"""
Step-Printing & Temporal Cadence Engine.
Calculates undercranking capture windows, exposure duration, shutter integration,
and step-printing frame repetition / decimation / slow-motion stretching.
Uses a unified global shutter curve across multi-frame exposure windows with dense
continuous sub-frame forward-splatting to eliminate discrete ghost frames.
"""

import math
from typing import List, Tuple, Dict, Any
import torch
import torch.nn.functional as F

try:
    from .flow_engine import FlowEngine
    from .vector_blur import forward_splat_substep, generate_shutter_weights
    from .occlusion import compute_occlusion_mask
except (ImportError, ValueError):
    from core.flow_engine import FlowEngine
    from core.vector_blur import forward_splat_substep, generate_shutter_weights
    from core.occlusion import compute_occlusion_mask


class StepPrinter:
    """
    Processes video frame tensors into the Wong Kar-Wai step-printed look.
    """

    @classmethod
    def process(
        cls,
        frames: torch.Tensor,
        input_fps: float = 24.0,
        target_capture_fps: float = 6.0,
        shutter_angle: float = 360.0,
        optical_flow_method: str = "RAFT-Small (Deep Learning)",
        flow_samples: int = 16,
        occlusion_aware: bool = True,
        occlusion_threshold: float = 1.5,
        vector_blur_intensity: float = 1.0,
        shutter_curve: str = "trailing_decay",
        cadence_mode: str = "step_print_repeat",
        step_repeat_count: int = 4,
        device: str = "auto"
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        frames: [T, C, H, W] in range [0, 1]
        Returns:
            output_frames: [T_out, C, H, W]
            output_fps: float
            occlusion_masks: [T_out, 1, H, W]
        """
        T, C, H, W = frames.shape
        target_device = FlowEngine.get_device(device)
        frames = frames.to(target_device)

        if T <= 1:
            dummy_mask = torch.ones((T, 1, H, W), device=target_device, dtype=frames.dtype)
            return frames, input_fps, dummy_mask

        # Ratio of input frames per capture interval
        window_size_float = max(1.0, float(input_fps) / max(0.1, float(target_capture_fps)))
        # Fraction of capture interval during which the shutter is open
        exposure_fraction = max(0.01, float(shutter_angle) / 360.0)
        # Number of input frame intervals to expose
        exposure_frames_float = max(1.0, window_size_float * exposure_fraction * vector_blur_intensity)

        # Precompute adjacent inter-frame optical flows
        # flows_fwd[k] is flow from frame k -> k+1
        # flows_bwd[k] is flow from frame k+1 -> k
        flows_fwd = []
        flows_bwd = []
        pair_masks = []

        for k in range(T - 1):
            f1 = frames[k:k+1]
            f2 = frames[k+1:k+2]

            flow_fwd, flow_bwd = FlowEngine.compute_bidirectional_flow(
                f1, f2, method=optical_flow_method, device=target_device
            )
            flows_fwd.append(flow_fwd)
            flows_bwd.append(flow_bwd)

            if occlusion_aware:
                m = compute_occlusion_mask(flow_fwd, flow_bwd, threshold=occlusion_threshold)
            else:
                m = torch.ones((1, 1, H, W), device=target_device, dtype=frames.dtype)
            pair_masks.append(m)

        # Append identity flow for last frame boundary
        zero_flow = torch.zeros((1, 2, H, W), device=target_device, dtype=frames.dtype)
        flows_fwd.append(zero_flow)
        flows_bwd.append(zero_flow)
        pair_masks.append(torch.ones((1, 1, H, W), device=target_device, dtype=frames.dtype))

        # Synthesize undercranked capture frames using UNIFIED global shutter exposure
        num_captures = max(1, int(math.ceil(T / window_size_float)))
        captured_frames = []
        captured_masks = []

        # Number of dense temporal samples across the entire exposure window
        sub_samples_per_window = max(12, int(round(window_size_float * max(4, flow_samples // 2))))

        for c_idx in range(num_captures):
            t_start = c_idx * window_size_float
            t_end = min(float(T - 1), t_start + exposure_frames_float)

            if t_start >= T:
                break

            if t_end <= t_start:
                idx = min(int(round(t_start)), T - 1)
                captured_frames.append(frames[idx:idx+1])
                captured_masks.append(pair_masks[min(idx, T - 1)])
                continue

            accum_f = torch.zeros_like(frames[0:1])
            accum_m = torch.zeros_like(pair_masks[0])
            total_w = 0.0

            # Generate dense continuous sub-samples across [t_start, t_end]
            for s_idx in range(sub_samples_per_window):
                # Normalized coordinate along the entire exposure window in [0, 1]
                tau_norm = float(s_idx) / float(sub_samples_per_window - 1)

                # Continuous time in frame indices
                tau_frame = t_start + tau_norm * (t_end - t_start)
                tau_frame = min(float(T - 1), max(0.0, tau_frame))

                # Integer frame segment
                k = min(T - 2, max(0, int(math.floor(tau_frame))))
                k_next = min(T - 1, k + 1)
                alpha = tau_frame - float(k)
                alpha = min(1.0, max(0.0, alpha))

                # Global shutter weight: STRICTLY ONE continuous curve across the whole window!
                if shutter_curve == "trailing_decay":
                    # Current frame (end of exposure) is sharpest, trails fade behind
                    w = math.exp(-2.2 * (1.0 - tau_norm))
                elif shutter_curve == "gaussian":
                    w = math.exp(-0.5 * ((tau_norm - 0.5) / 0.28) ** 2)
                elif shutter_curve == "triangle":
                    w = max(0.05, 1.0 - 2.0 * abs(tau_norm - 0.5))
                else:
                    w = 1.0

                # Continuous sub-frame via forward splatting
                if k == k_next or alpha < 1e-4:
                    sub_f = frames[k:k+1]
                elif alpha > 0.999:
                    sub_f = frames[k_next:k_next+1]
                else:
                    sub_f = forward_splat_substep(
                        img1=frames[k:k+1],
                        img2=frames[k_next:k_next+1],
                        flow_fwd=flows_fwd[k],
                        flow_bwd=flows_bwd[k],
                        s=alpha,
                        occlusion_aware=occlusion_aware,
                        occlusion_threshold=occlusion_threshold
                    )

                accum_f += sub_f * w
                accum_m += pair_masks[k] * w
                total_w += w

            captured_frames.append(accum_f / max(1e-5, total_w))
            captured_masks.append(accum_m / max(1e-5, total_w))

        if len(captured_frames) == 0:
            captured_frames.append(frames[0:1])
            captured_masks.append(pair_masks[0])

        # Step printing output cadence assembly
        out_frames_list = []
        out_masks_list = []

        if cadence_mode == "decimate_to_target_fps":
            out_frames_list = captured_frames
            out_masks_list = captured_masks
            out_fps = float(target_capture_fps)

        elif cadence_mode == "slow_motion_stretch":
            repeat_k = max(1, int(step_repeat_count))
            for cf, cm in zip(captured_frames, captured_masks):
                for _ in range(repeat_k):
                    out_frames_list.append(cf)
                    out_masks_list.append(cm)
            out_fps = float(input_fps)

        else:
            # step_print_repeat
            repeat_k = max(1, int(round(window_size_float)))
            for cf, cm in zip(captured_frames, captured_masks):
                for _ in range(repeat_k):
                    if len(out_frames_list) < T:
                        out_frames_list.append(cf)
                        out_masks_list.append(cm)

            while len(out_frames_list) < T:
                out_frames_list.append(captured_frames[-1])
                out_masks_list.append(captured_masks[-1])
            out_frames_list = out_frames_list[:T]
            out_masks_list = out_masks_list[:T]
            out_fps = float(input_fps)

        final_frames = torch.cat(out_frames_list, dim=0)
        final_masks = torch.cat(out_masks_list, dim=0)
        return final_frames, out_fps, final_masks
