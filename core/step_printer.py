"""
Step-Printing & Temporal Cadence Engine.
Calculates undercranking capture windows, exposure duration, shutter integration,
and step-printing frame repetition / decimation / slow-motion stretching.
"""

import math
from typing import List, Tuple, Dict, Any
import torch
import torch.nn.functional as F

try:
    from .flow_engine import FlowEngine
    from .vector_blur import integrate_vector_path
except (ImportError, ValueError):
    from core.flow_engine import FlowEngine
    from core.vector_blur import integrate_vector_path


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
        flow_samples: int = 12,
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
        exposure_frames_float = max(1.0, window_size_float * exposure_fraction)

        # Precompute adjacent inter-frame optical flows and vector-blurred sub-segments
        # Segment k is the vector-blurred transition from frames[k] to frames[k+1]
        segments = []
        masks = []

        # Effective shutter fraction per adjacent frame transition
        shutter_per_step = min(2.0, exposure_fraction * vector_blur_intensity)

        for k in range(T - 1):
            f1 = frames[k:k+1]
            f2 = frames[k+1:k+2]

            flow_fwd, flow_bwd = FlowEngine.compute_bidirectional_flow(
                f1, f2, method=optical_flow_method, device=target_device
            )

            seg = integrate_vector_path(
                f1, f2,
                flow_fwd=flow_fwd,
                flow_bwd=flow_bwd,
                shutter_fraction=shutter_per_step,
                num_samples=flow_samples,
                shutter_curve=shutter_curve,
                occlusion_aware=occlusion_aware,
                occlusion_threshold=occlusion_threshold
            )
            segments.append(seg)

            if occlusion_aware:
                from .occlusion import compute_occlusion_mask
                m = compute_occlusion_mask(flow_fwd, flow_bwd, threshold=occlusion_threshold)
            else:
                m = torch.ones((1, 1, H, W), device=target_device, dtype=frames.dtype)
            masks.append(m)

        # Append last frame as static segment
        segments.append(frames[-1:])
        masks.append(torch.ones((1, 1, H, W), device=target_device, dtype=frames.dtype))

        # Now synthesize undercranked capture frames
        # Each capture frame i starts at time t_start = i * window_size_float
        num_captures = max(1, int(math.ceil(T / window_size_float)))
        captured_frames = []
        captured_masks = []

        for c_idx in range(num_captures):
            t_start = c_idx * window_size_float
            t_end = min(float(T - 1), t_start + exposure_frames_float)

            idx_start = int(math.floor(t_start))
            idx_end = min(T - 1, int(math.ceil(t_end)))

            if idx_start >= T:
                break

            if idx_start == idx_end or idx_end <= idx_start:
                captured_frames.append(segments[min(idx_start, T - 1)])
                captured_masks.append(masks[min(idx_start, T - 1)])
                continue

            # Multi-frame exposure accumulation across [idx_start, idx_end]
            accum_f = torch.zeros_like(frames[0:1])
            accum_m = torch.zeros_like(masks[0])
            total_w = 0.0

            sub_range = list(range(idx_start, min(idx_end + 1, T)))
            count = len(sub_range)

            for step_i, k in enumerate(sub_range):
                # Weight by position in exposure window
                if count > 1:
                    norm_pos = step_i / float(count - 1)
                else:
                    norm_pos = 1.0

                if shutter_curve == "trailing_decay":
                    w = math.exp(-2.0 * (1.0 - norm_pos))
                elif shutter_curve == "gaussian":
                    w = math.exp(-0.5 * ((norm_pos - 0.5) / 0.3) ** 2)
                elif shutter_curve == "triangle":
                    w = 1.0 - 2.0 * abs(norm_pos - 0.5)
                else:
                    w = 1.0

                accum_f += segments[k] * w
                accum_m += masks[k] * w
                total_w += w

            captured_frames.append(accum_f / max(1e-5, total_w))
            captured_masks.append(accum_m / max(1e-5, total_w))

        if len(captured_frames) == 0:
            captured_frames.append(frames[0:1])
            captured_masks.append(masks[0:1])

        # Step printing output cadence assembly
        out_frames_list = []
        out_masks_list = []

        if cadence_mode == "decimate_to_target_fps":
            # True low-fps output (each unique capture frame emitted once)
            out_frames_list = captured_frames
            out_masks_list = captured_masks
            out_fps = float(target_capture_fps)

        elif cadence_mode == "slow_motion_stretch":
            # Stretch each capture frame by step_repeat_count to produce slow motion
            repeat_k = max(1, int(step_repeat_count))
            for cf, cm in zip(captured_frames, captured_masks):
                for _ in range(repeat_k):
                    out_frames_list.append(cf)
                    out_masks_list.append(cm)
            out_fps = float(input_fps)

        else:
            # "step_print_repeat" (standard Wong Kar-Wai cadence matching input timeline duration)
            # Total output frames should match input frame count T
            repeat_k = max(1, int(round(window_size_float)))
            for cf, cm in zip(captured_frames, captured_masks):
                for _ in range(repeat_k):
                    if len(out_frames_list) < T:
                        out_frames_list.append(cf)
                        out_masks_list.append(cm)

            # Pad or trim if floating point rounding differed slightly from T
            while len(out_frames_list) < T:
                out_frames_list.append(captured_frames[-1])
                out_masks_list.append(captured_masks[-1])
            out_frames_list = out_frames_list[:T]
            out_masks_list = out_masks_list[:T]
            out_fps = float(input_fps)

        final_frames = torch.cat(out_frames_list, dim=0)
        final_masks = torch.cat(out_masks_list, dim=0)
        return final_frames, out_fps, final_masks
