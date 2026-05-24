"""
Evaluator module for synthetic data evaluation.

Provides CLIP-based evaluation pipeline including:
- Zero-shot classification
- Feature extraction and classifier training
- Results management
"""

from .feature_extractor import CLIPFeatureExtractor
from .clip_evaluator import (
    CLIPClassifierTrainer,
    CLIPMLPClassifierTrainer,
    CLIPZeroShotEvaluator,
)
from .data_loader import SyntheticDataset, SyntheticDirDataset, RealDataset
from .results_manager import EvaluationResultsManager

__all__ = [
    'CLIPFeatureExtractor',
    'CLIPClassifierTrainer',
    'CLIPMLPClassifierTrainer',
    'CLIPZeroShotEvaluator',
    'SyntheticDataset',
    'SyntheticDirDataset',
    'RealDataset',
    'EvaluationResultsManager',
]

