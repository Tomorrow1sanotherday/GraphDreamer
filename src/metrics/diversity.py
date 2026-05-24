"""Structural diversity metrics for scene graph collections."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union


JsonDict = Dict[str, Any]
JsonLike = Union[JsonDict, Sequence[Any]]


@dataclass(frozen=True)
class DiversityReport:
    """Holds per-component entropy values and aggregate diversity."""

    subject_entropy: float
    object_entropy: float
    attribute_entropy: float
    relation_entropy: float
    subject_counts: Dict[str, int]
    object_counts: Dict[str, int]
    attribute_counts: Dict[str, int]
    relation_counts: Dict[str, int]

    @property
    def total_diversity(self) -> float:
        """Return the sum of enabled entropy components."""
        return (
            self.subject_entropy
            + self.object_entropy
            + self.attribute_entropy
            + self.relation_entropy
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report into a JSON-serializable dictionary."""
        return {
            "diversity": self.total_diversity,
            "entropies": {
                "subject": self.subject_entropy,
                "object": self.object_entropy,
                "attribute": self.attribute_entropy,
                "relation": self.relation_entropy,
            },
            "counts": {
                "subject": self.subject_counts,
                "object": self.object_counts,
                "attribute": self.attribute_counts,
                "relation": self.relation_counts,
            },
        }


class StructuralDiversityCalculator:
    """Compute structural diversity over scene graph collections."""

    def __init__(self, log_base: str = "e") -> None:
        """
        Initialize a diversity calculator.

        Args:
            log_base: Entropy log base. Supported values: "e", "2", "10".
        """
        valid_bases = {"e", "2", "10"}
        if log_base not in valid_bases:
            raise ValueError(f"log_base must be one of {valid_bases}, got: {log_base}")
        self.log_base = log_base

    def compute_from_file(self, json_path: Union[str, Path]) -> DiversityReport:
        """Load a JSON file and compute structural diversity."""
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return self.compute(data)

    def compute(self, data: JsonLike) -> DiversityReport:
        """Compute diversity statistics from parsed JSON content."""
        subject_counter: Counter[str] = Counter()
        object_counter: Counter[str] = Counter()
        attribute_counter: Counter[str] = Counter()
        relation_counter: Counter[str] = Counter()

        for scene_graph in self._iter_scene_graphs(data):
            subject = self._extract_subject(scene_graph)
            if subject:
                subject_counter[subject] += 1

            for obj in self._extract_objects(scene_graph):
                object_counter[obj] += 1

            for relation in self._extract_relations(scene_graph):
                relation_counter[relation] += 1

            for attribute in self._extract_attributes(scene_graph):
                attribute_counter[attribute] += 1

        return DiversityReport(
            subject_entropy=self._entropy(subject_counter),
            object_entropy=self._entropy(object_counter),
            attribute_entropy=self._entropy(attribute_counter),
            relation_entropy=self._entropy(relation_counter),
            subject_counts=dict(subject_counter),
            object_counts=dict(object_counter),
            attribute_counts=dict(attribute_counter),
            relation_counts=dict(relation_counter),
        )

    def _iter_scene_graphs(self, data: JsonLike) -> Iterable[JsonDict]:
        """
        Yield scene graph dictionaries from common dataset layouts.

        Supported layouts:
            - {"results": [{"scene_graph": {...}}, ...]}
            - {"results": [{...scene_graph...}, ...]}
            - [{"scene_graph": {...}}, ...]
            - [{...scene_graph...}, ...]
        """
        items: Iterable[Any]
        if isinstance(data, Mapping):
            items = data.get("results", [])
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            items = data
        else:
            items = []

        for item in items:
            if not isinstance(item, Mapping):
                continue
            scene_graph = item.get("scene_graph", item)
            if isinstance(scene_graph, Mapping):
                yield dict(scene_graph)

    def _extract_subject(self, scene_graph: JsonDict) -> Optional[str]:
        subject = scene_graph.get("subject")
        token = self._coerce_token(subject)
        return token if token else None

    def _extract_objects(self, scene_graph: JsonDict) -> Iterable[str]:
        objects = scene_graph.get("objects", [])
        if not isinstance(objects, Sequence):
            return []
        return [token for token in (self._coerce_token(obj) for obj in objects) if token]

    def _extract_relations(self, scene_graph: JsonDict) -> Iterable[str]:
        relations = scene_graph.get("relations", [])
        if not isinstance(relations, Sequence):
            return []
        return [token for token in (self._coerce_token(rel) for rel in relations) if token]

    def _extract_attributes(self, scene_graph: JsonDict) -> Iterable[str]:
        attributes = scene_graph.get("attributes", {})
        return list(self._flatten_attributes(attributes))

    def _flatten_attributes(self, value: Any, key_prefix: str = "") -> Iterable[str]:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                next_prefix = f"{key_prefix}.{key}" if key_prefix else str(key)
                yield from self._flatten_attributes(nested, next_prefix)
            return

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for element in value:
                yield from self._flatten_attributes(element, key_prefix)
            return

        token = self._coerce_token(value)
        if not token:
            return
        yield f"{key_prefix}:{token}" if key_prefix else token

    def _coerce_token(self, value: Any) -> str:
        """
        Convert heterogeneous field content into a normalized token.

        Priority for mappings:
            name -> relation -> value -> label
        """
        if isinstance(value, Mapping):
            for key in ("name", "relation", "value", "label"):
                if key in value:
                    return self._coerce_token(value.get(key))
            return ""

        if value is None:
            return ""
        token = str(value).strip()
        return token

    def _entropy(self, counter: Counter[str]) -> float:
        total = sum(counter.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counter.values():
            probability = count / total
            if probability <= 0:
                continue
            entropy -= probability * self._log(probability)
        return entropy

    def _log(self, value: float) -> float:
        if self.log_base == "2":
            return math.log2(value)
        if self.log_base == "10":
            return math.log10(value)
        return math.log(value)


__all__ = ["DiversityReport", "StructuralDiversityCalculator"]

if __name__ == "__main__":
    calculator = StructuralDiversityCalculator()
    report = calculator.compute_from_file("/mnt/sda/runhaofu/DivSyn/data/cub2011/scene_graphs/co_object_10/scene_graphs_semantically_programmatic_1_1.json")
    print(report.to_dict())