"""
ComfyUI-WKW-StepPrinting
Wong Kar-Wai / Christopher Doyle Step-Printing & Deep Optical Flow Motion Blur Node Pack.
"""

try:
    from .nodes.wkw_step_printing_node import WKWStepPrintingNode
    from .nodes.wkw_grading_node import WKWGradingNode
    from .nodes.wkw_vector_viz_node import WKWVectorVisualizerNode
except (ImportError, ValueError):
    from nodes.wkw_step_printing_node import WKWStepPrintingNode
    from nodes.wkw_grading_node import WKWGradingNode
    from nodes.wkw_vector_viz_node import WKWVectorVisualizerNode

NODE_CLASS_MAPPINGS = {
    "WKWStepPrintingNode": WKWStepPrintingNode,
    "WKWGradingNode": WKWGradingNode,
    "WKWVectorVisualizerNode": WKWVectorVisualizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WKWStepPrintingNode": "🎬 WKW Step Printing (Optical Flow & Occlusion)",
    "WKWGradingNode": "🏮 WKW Film & Lens Style (Neon Bloom / Halation)",
    "WKWVectorVisualizerNode": "👁️ WKW Motion Vector Visualizer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
