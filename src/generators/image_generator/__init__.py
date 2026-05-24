"""
Image Generator Module

This module provides functionality to generate images from scene graphs
using Text-to-Image diffusion models (e.g., Stable Diffusion XL).
"""

from .data_types import (
    ObjectEntity,
    Relation,
    SceneGraph,
    SceneGraphItem,
    ImageGenerationTask,
    ImageGenerationResult,
    SyntheticImageDataset,
    GeneratorConfig
)

from .generator import (
    BaseImageGenerator,
    DiffusionImageGenerator,
    ImageGenerationManager,
    StreamingJsonWriter,
    generate_images_batch
)

__all__ = [
    # Data types
    "ObjectEntity",
    "Relation",
    "SceneGraph",
    "SceneGraphItem",
    "ImageGenerationTask",
    "ImageGenerationResult",
    "SyntheticImageDataset",
    "GeneratorConfig",
    # Generator classes
    "BaseImageGenerator",
    "DiffusionImageGenerator",
    "ImageGenerationManager",
    "StreamingJsonWriter",
    # Functions
    "generate_images_batch"
]

