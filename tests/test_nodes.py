"""
Unit and Integration Tests for ComfyUI-WKW-StepPrinting.
"""

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.flow_engine import FlowEngine
from core.occlusion import compute_occlusion_mask, compute_disocclusion_mask, backward_warp
from core.vector_blur import integrate_vector_path, generate_shutter_weights
from core.step_printer import StepPrinter
from nodes.wkw_step_printing_node import WKWStepPrintingNode
from nodes.wkw_vector_viz_node import WKWVectorVisualizerNode


class TestWKWStepPrinting(unittest.TestCase):

    def setUp(self):
        self.T, self.H, self.W, self.C = 12, 128, 128, 3
        self.frames_hwc = torch.zeros((self.T, self.H, self.W, self.C), dtype=torch.float32)
        for t in range(self.T):
            pos_y = 20 + t * 4
            pos_x = 20 + t * 6
            self.frames_hwc[t, pos_y:pos_y+24, pos_x:pos_x+24, :] = 1.0
            self.frames_hwc[t, 10:25, 90:110, 0] = 0.9
            self.frames_hwc[t, 10:25, 90:110, 1] = 0.8
            self.frames_hwc[t, 10:25, 90:110, 2] = 0.2

    def test_flow_engine_and_occlusion(self):
        img1 = self.frames_hwc[0:1].permute(0, 3, 1, 2)
        img2 = self.frames_hwc[1:2].permute(0, 3, 1, 2)

        flow_fwd, flow_bwd = FlowEngine.compute_bidirectional_flow(img1, img2, method="RAFT-Small (Deep Learning)")
        self.assertEqual(flow_fwd.shape, (1, 2, self.H, self.W))
        self.assertEqual(flow_bwd.shape, (1, 2, self.H, self.W))

        occ = compute_occlusion_mask(flow_fwd, flow_bwd)
        disocc = compute_disocclusion_mask(flow_fwd, flow_bwd)
        self.assertEqual(occ.shape, (1, 1, self.H, self.W))
        self.assertEqual(disocc.shape, (1, 1, self.H, self.W))
        self.assertTrue(torch.all(occ >= 0.0) and torch.all(occ <= 1.0))
        self.assertTrue(torch.all(disocc >= 0.0) and torch.all(disocc <= 1.0))

    def test_vector_path_integration(self):
        img1 = self.frames_hwc[0:1].permute(0, 3, 1, 2)
        img2 = self.frames_hwc[1:2].permute(0, 3, 1, 2)
        flow_fwd, flow_bwd = FlowEngine.compute_bidirectional_flow(img1, img2, method="RAFT-Small (Deep Learning)")

        for curve in ["trailing_decay", "gaussian", "box", "triangle"]:
            blurred = integrate_vector_path(
                img1, img2, flow_fwd, flow_bwd,
                shutter_fraction=1.0, num_samples=8,
                shutter_curve=curve, occlusion_aware=True
            )
            self.assertEqual(blurred.shape, (1, 3, self.H, self.W))
            self.assertTrue(torch.all(blurred >= 0.0) and torch.all(blurred <= 1.0))

    def test_step_printing_cadence_modes(self):
        node = WKWStepPrintingNode()

        # Mode 1: step_print_repeat
        out_imgs, out_fps, out_masks = node.apply_step_printing(
            images=self.frames_hwc,
            input_fps=24.0,
            target_capture_fps=6.0,
            shutter_angle=360.0,
            shutter_timing="trailing (past trail only)",
            optical_flow_method="RAFT-Small (Deep Learning)",
            flow_samples=6,
            occlusion_aware=True,
            occlusion_threshold=1.5,
            shutter_curve="trailing_decay",
            cadence_mode="step_print_repeat",
            step_repeat_count=4,
            vector_blur_intensity=1.0,
            device="auto"
        )
        self.assertEqual(out_imgs.shape, (self.T, self.H, self.W, self.C))
        self.assertEqual(out_masks.shape, (self.T, self.H, self.W))
        self.assertEqual(out_fps, 24.0)

        # Mode 2: decimate_to_target_fps
        out_imgs2, out_fps2, _ = node.apply_step_printing(
            images=self.frames_hwc,
            input_fps=24.0,
            target_capture_fps=6.0,
            shutter_angle=360.0,
            shutter_timing="trailing (past trail only)",
            optical_flow_method="RAFT-Small (Deep Learning)",
            flow_samples=6,
            occlusion_aware=True,
            occlusion_threshold=1.5,
            shutter_curve="trailing_decay",
            cadence_mode="decimate_to_target_fps",
            step_repeat_count=4,
            vector_blur_intensity=1.0,
            device="auto"
        )
        self.assertEqual(out_imgs2.shape[0], 3)
        self.assertEqual(out_fps2, 6.0)

        # Mode 3: slow_motion_stretch
        out_imgs3, out_fps3, _ = node.apply_step_printing(
            images=self.frames_hwc,
            input_fps=24.0,
            target_capture_fps=6.0,
            shutter_angle=360.0,
            shutter_timing="trailing (past trail only)",
            optical_flow_method="RAFT-Small (Deep Learning)",
            flow_samples=6,
            occlusion_aware=True,
            occlusion_threshold=1.5,
            shutter_curve="trailing_decay",
            cadence_mode="slow_motion_stretch",
            step_repeat_count=4,
            vector_blur_intensity=1.0,
            device="auto"
        )
        self.assertEqual(out_imgs3.shape[0], 12)
        self.assertEqual(out_fps3, 24.0)

    def test_vector_visualizer_node(self):
        viz = WKWVectorVisualizerNode()
        flow_viz, occ_m, disocc_m = viz.visualize(
            images=self.frames_hwc[:3],
            optical_flow_method="DIS (Fast OpenCV)",
            occlusion_threshold=1.5,
            device="auto"
        )
        self.assertEqual(flow_viz.shape, (3, self.H, self.W, 3))
        self.assertEqual(occ_m.shape, (3, self.H, self.W, 3))
        self.assertEqual(disocc_m.shape, (3, self.H, self.W, 3))


if __name__ == "__main__":
    unittest.main()
