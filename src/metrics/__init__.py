"""Scene graph quality metrics: diversity, realism, and subset selection."""

from src.metrics.diversity import DiversityReport, StructuralDiversityCalculator
from src.metrics.selection import GreedySceneGraphSelector, SelectionResult

__all__ = [
    "DiversityReport",
    "StructuralDiversityCalculator",
    "GreedySceneGraphSelector",
    "SelectionResult",
]

try:
    from src.metrics.realism import RealismScorer, RealismResult, DimensionScore
    __all__.extend(["RealismScorer", "RealismResult", "DimensionScore"])
except ImportError:
    pass
