# Sionna Ray Tracing Integration for HY-Motion
# This module provides integration between HY-Motion's LLM-based motion generation
# and NVIDIA Sionna's ray tracing capabilities for wireless channel simulation.
#
# Features:
# - Convert HY-Motion generated SMPL meshes to Sionna-compatible format
# - Build ray tracing scenes (indoor/outdoor) with human bodies
# - Run ray tracing simulations with configurable RF parameters
# - Interactive 3D visualization with Three.js
# - Static visualization with matplotlib
#
# Example:
#     from hymotion.sionna import SionnaRayVisualizer, RayTracingConfig
#     from hymotion.utils.t2m_runtime import T2MRuntime
#
#     # Initialize
#     runtime = T2MRuntime(config_path="path/to/config.yml")
#     config = RayTracingConfig(frequency=3.5e9)
#     visualizer = SionnaRayVisualizer(config)
#
#     # Generate motion and visualize
#     results = visualizer.generate_and_visualize(
#         runtime=runtime,
#         text_prompt="A person walking forward",
#         duration=3.0,
#     )
 
from .ray_visualization import (
    SionnaRayVisualizer,
    MotionToSionnaConverter,
    RayTracingConfig,
    visualize_motion_ray_tracing,
)
from .scene_builder import (
    SionnaSceneBuilder,
    MaterialProperties,
    SceneObject,
    TransceiverConfig,
    create_indoor_scene,
    create_outdoor_scene,
)
from .interactive_viz import (
    InteractiveRayVisualizer,
    create_html_visualization,
)
from .wireless_sensing import (
    WirelessSensingProcessor,
    WirelessSensingConfig,
    MotionWirelessDataset,
    compute_signal_features,
    compare_signal_representations,
)
from .signal_visualization import (
    SignalVisualizer,
    create_summary_report,
)
 
__all__ = [
    # Ray Visualization
    "SionnaRayVisualizer",
    "MotionToSionnaConverter",
    "RayTracingConfig",
    "visualize_motion_ray_tracing",
    # Scene Builder
    "SionnaSceneBuilder",
    "MaterialProperties",
    "SceneObject",
    "TransceiverConfig",
    "create_indoor_scene",
    "create_outdoor_scene",
    # Interactive Visualization
    "InteractiveRayVisualizer",
    "create_html_visualization",
    # Wireless Sensing
    "WirelessSensingProcessor",
    "WirelessSensingConfig",
    "MotionWirelessDataset",
    "compute_signal_features",
    "compare_signal_representations",
    # Signal Visualization
    "SignalVisualizer",
    "create_summary_report",
]