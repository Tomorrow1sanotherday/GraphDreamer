"""
Reality QA Generator Module

Generates feasibility-checking questions from scene graphs to evaluate
whether node combinations are physically / semantically plausible.

For each scene graph g, the reality score is defined as:

    R(g) = 1/2 * (R_SA + R_SO)

where each component is obtained by querying an LLM as a zero-shot scorer:

    R_SA    — Subject–Attribute:          "Can <S> be <A>?"
    R_SO    — Subject–Object:             "Can <S> be found in or near <O>?"

Additional dimensions:

    R_SRO   — Subject–Relation–Object:    "Can <S> be <R> <O>?"
    R_SCENE — Holistic scene:             "Is it realistic to see <S> <A> <R> <O>?"
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


QUESTION_TEMPLATES: Dict[str, str] = {
    "R_SA":    "Can a {subject} be {attribute}?",
    "R_SO":    "Can a {subject} be found in or near {object}?",
    "R_SRO":   "Can a {subject} be {relation} {object}?",
    "R_SCENE": "Is it realistic to see a {subject} {attribute} {relation} {object}?",
}


@dataclass
class RealityQuestion:
    """A single reality-check question with metadata."""
    id: int
    dimension: str
    question: str
    subject: str
    attribute: str = ""
    object: str = ""
    relation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "dimension": self.dimension,
            "question": self.question,
            "subject": self.subject,
        }
        if self.attribute:
            d["attribute"] = self.attribute
        if self.object:
            d["object"] = self.object
        if self.relation:
            d["relation"] = self.relation
        return d


@dataclass
class RealityQARecord:
    """Reality QA output for one scene graph."""
    id: int
    label: int
    label_name: str
    scene_graph: Dict[str, Any]
    caption: str
    reality_qa: List[RealityQuestion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "label_name": self.label_name,
            "scene_graph": self.scene_graph,
            "caption": self.caption,
            "reality_qa": [q.to_dict() for q in self.reality_qa],
        }


class RealityQAGenerator:
    """
    Deterministic reality-check question generator from scene graphs.

    For each scene graph, produces questions across three dimensions
    (R_SA, R_SO, R_SRO) that can be scored by a zero-shot LLM.
    """

    def generate(self, scene_graph: Dict[str, Any]) -> List[RealityQuestion]:
        """
        Generate reality QA questions from a single scene graph.

        Args:
            scene_graph: A scene graph dict with keys:
                subject, objects, relations, attributes.

        Returns:
            List of RealityQuestion objects.
        """
        subject = scene_graph["subject"]["name"]
        objects = [o["name"] for o in scene_graph["objects"]]
        attrs = list(scene_graph.get("attributes", {}).values())
        obj_map = {o["id"]: o["name"] for o in scene_graph["objects"]}

        qs: List[RealityQuestion] = []
        qid = 1

        for attr in attrs:
            qs.append(RealityQuestion(
                id=qid, dimension="R_SA",
                question=QUESTION_TEMPLATES["R_SA"].format(subject=subject, attribute=attr),
                subject=subject, attribute=attr,
            ))
            qid += 1

        for obj in objects:
            qs.append(RealityQuestion(
                id=qid, dimension="R_SO",
                question=QUESTION_TEMPLATES["R_SO"].format(subject=subject, object=obj),
                subject=subject, object=obj,
            ))
            qid += 1

        for rel in scene_graph.get("relations", []):
            obj_name = obj_map.get(rel["object_id"])
            if obj_name:
                qs.append(RealityQuestion(
                    id=qid, dimension="R_SRO",
                    question=QUESTION_TEMPLATES["R_SRO"].format(
                        subject=subject, relation=rel["relation"], object=obj_name,
                    ),
                    subject=subject, relation=rel["relation"], object=obj_name,
                ))
                qid += 1

        for rel in scene_graph.get("relations", []):
            obj_name = obj_map.get(rel["object_id"])
            if obj_name and attrs:
                qs.append(RealityQuestion(
                    id=qid, dimension="R_SCENE",
                    question=QUESTION_TEMPLATES["R_SCENE"].format(
                        subject=subject, attribute=attrs[0],
                        relation=rel["relation"], object=obj_name,
                    ),
                    subject=subject, attribute=attrs[0],
                    relation=rel["relation"], object=obj_name,
                ))
                qid += 1

        return qs

    def process(self, input_path: str, output_path: str) -> List[RealityQARecord]:
        """
        Process an entire scene-graph JSON file and write reality QA output.

        Args:
            input_path: Path to the input scene-graph JSON.
            output_path: Path for the output reality-QA JSON.

        Returns:
            List of RealityQARecord objects.
        """
        in_path = Path(input_path)
        out_path = Path(output_path)

        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data['results'])} scene graphs from {in_path}")

        records: List[RealityQARecord] = []
        for item in data["results"]:
            qa = self.generate(item["scene_graph"])
            records.append(RealityQARecord(
                id=item["id"],
                label=item["label"],
                label_name=item["label_name"],
                scene_graph=item["scene_graph"],
                caption=item.get("caption", ""),
                reality_qa=qa,
            ))

        output = {
            "dataset": data["dataset"],
            "total": len(records),
            "description": (
                "Reality QA for scene-graph feasibility assessment. "
                "R(g) = 1/2 * (R_SA + R_SO). "
                "Each question is designed for zero-shot LLM scoring."
            ),
            "results": [r.to_dict() for r in records],
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(records)} reality QA records to {out_path}")
        return records
