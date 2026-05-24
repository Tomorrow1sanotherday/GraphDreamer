"""CopulaSyn v2 source package."""

from .api import StreamGenerator
from .generators.coobject_generator import ObjectCooccurrenceGenerator
from .generators.image_generator import (
    GeneratorConfig,
    DiffusionImageGenerator,
    ImageGenerationManager,
    generate_images_batch
)
from .evaluator import (
    CLIPFeatureExtractor,
    CLIPClassifierTrainer,
    CLIPZeroShotEvaluator,
    SyntheticDataset,
    RealDataset,
    EvaluationResultsManager,
)

__all__ = [
    "StreamGenerator",
    "ObjectCooccurrenceGenerator",
    # Image Generator
    "GeneratorConfig",
    "DiffusionImageGenerator",
    "ImageGenerationManager",
    "generate_images_batch",
    # Evaluator
    "CLIPFeatureExtractor",
    "CLIPClassifierTrainer",
    "CLIPZeroShotEvaluator",
    "SyntheticDataset",
    "RealDataset",
    "EvaluationResultsManager",
]

