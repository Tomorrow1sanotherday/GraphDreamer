"""
Structural Realism Scorer

Evaluates scene-graph feasibility by querying an LLM as a zero-shot scorer.
For each scene graph g the realism score is:

    R(g) = mean(R_SA, R_SO, R_SRO, R_SCENE)

Dimensions:
    R_SA    — Subject–Attribute          "Can <S> be <A>?"
    R_SO    — Subject–Object             "Can <S> be found in or near <O>?"
    R_SRO   — Subject–Relation–Object    "Can <S> be <R> <O>?"
    R_SCENE — Holistic scene             "Is it realistic to see <S> <A> <R> <O>?"

The LLM returns a soft score in [0, 1] for each question.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCORING_MODES = ("default", "structural")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a plausibility scorer. "
    "For each question, rate how realistic the described scenario is "
    "on a scale from 0.0 to 1.0:\n"
    "  0.0 = completely impossible\n"
    "  0.5 = unusual but possible\n"
    "  1.0 = very common and natural\n"
    "Return ONLY a single decimal number (e.g. 0.7). No other text."
)

GROUPED_SYSTEM_PROMPT = (
    "You are a plausibility scorer. "
    "Score each structured scene item on a scale from 0.0 to 1.0:\n"
    "  0.0 = completely impossible\n"
    "  0.5 = unusual but possible\n"
    "  1.0 = very common and natural\n"
    "Dimension meanings:\n"
    "  R_SA: subject-attribute or subject-action compatibility\n"
    "  R_SO: subject-object compatibility\n"
    "  R_SRO: subject-relation-object compatibility\n"
    "  R_SCENE: holistic scene plausibility\n"
    "  R_AO: action-object compatibility\n"
    "Return ONLY a JSON array of decimal numbers in the same order as the items. "
    "Example: [0.8, 0.6, 1.0, 0.7]. No other text."
)

TEMPLATES: Dict[str, str] = {
    "R_SA":    "Can a {subject} be {attribute}?",
    "R_SO":    "Can a {subject} be found in or near {object}?",
    "R_SRO":   "Can a {subject} be {relation} {object}?",
    "R_SCENE": "Is it realistic to see a {subject} {attribute} {relation} {object}?",
}

TEMPLATES_STRUCTURAL: Dict[str, str] = {
    "R_SA": "Can a {subject} perform the action of {action}?",
    "R_SO": "Can a {subject} be found in or near {object}?",
    "R_AO": "Can the action of {action} occur in or near {object}?",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DimensionScore:
    """Score for a single realism dimension."""
    dimension: str
    question: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"dimension": self.dimension, "question": self.question, "score": self.score}


@dataclass
class RealismResult:
    """Aggregated realism result for one scene graph."""
    id: int
    label: int
    label_name: str
    realism_score: float
    dimension_scores: List[DimensionScore] = field(default_factory=list)
    scene_graph: Optional[Dict[str, Any]] = None
    caption: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "label": self.label,
            "label_name": self.label_name,
            "realism_score": round(self.realism_score, 4),
        }
        if self.scene_graph is not None:
            d["scene_graph"] = self.scene_graph
        if self.caption:
            d["caption"] = self.caption
        d["dimension_scores"] = [ds.to_dict() for ds in self.dimension_scores]
        return d


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class RealismScorer:
    """
    Score scene-graph realism via LLM zero-shot evaluation.

    Uses ``StreamGenerator`` from ``src.api`` for async batched LLM calls.
    """

    def __init__(self, stream_generator: Any) -> None:
        """
        Args:
            stream_generator: An instance of ``StreamGenerator``.
        """
        self.stream_gen = stream_generator

    # ----- question construction -----

    @staticmethod
    def build_question_specs(scene_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build ordered default-mode question specs from a scene graph."""
        subject = scene_graph["subject"]["name"]
        attrs = list(scene_graph.get("attributes", {}).values())
        objects = [o["name"] for o in scene_graph.get("objects", [])]
        relations = scene_graph.get("relations", [])
        obj_map = {o["id"]: o["name"] for o in scene_graph.get("objects", [])}

        specs: List[Dict[str, Any]] = []

        for attr in attrs:
            specs.append({
                "dimension": "R_SA",
                "question": TEMPLATES["R_SA"].format(subject=subject, attribute=attr),
                "fields": {"subject": subject, "attribute": attr},
            })

        for obj in objects:
            specs.append({
                "dimension": "R_SO",
                "question": TEMPLATES["R_SO"].format(subject=subject, object=obj),
                "fields": {"subject": subject, "object": obj},
            })

        for rel in relations:
            obj_name = obj_map.get(rel["object_id"])
            if obj_name:
                specs.append({
                    "dimension": "R_SRO",
                    "question": TEMPLATES["R_SRO"].format(
                        subject=subject,
                        relation=rel["relation"],
                        object=obj_name,
                    ),
                    "fields": {
                        "subject": subject,
                        "relation": rel["relation"],
                        "object": obj_name,
                    },
                })

        for rel in relations:
            obj_name = obj_map.get(rel["object_id"])
            if obj_name and attrs:
                specs.append({
                    "dimension": "R_SCENE",
                    "question": TEMPLATES["R_SCENE"].format(
                        subject=subject,
                        attribute=attrs[0],
                        relation=rel["relation"],
                        object=obj_name,
                    ),
                    "fields": {
                        "subject": subject,
                        "attribute": attrs[0],
                        "relation": rel["relation"],
                        "object": obj_name,
                    },
                })

        return specs

    @staticmethod
    def build_questions(scene_graph: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Build realism questions from a single scene graph.

        Returns:
            List of dicts with keys ``dimension`` and ``prompt``.
        """
        return [
            {"dimension": spec["dimension"], "prompt": spec["question"]}
            for spec in RealismScorer.build_question_specs(scene_graph)
        ]

    @staticmethod
    def build_question_specs_structural(scene_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build ordered structural-mode question specs from a scene graph.

        R(g) = (1/3)(R_SA + R_SO + R_AO)

        Dimensions:
            R_SA: "Can S perform A?"   (subject–action compatibility)
            R_SO: "Can S exist in O?"  (subject–object compatibility)
            R_AO: "Can A occur in O?"  (action–object compatibility)
        """
        subject = scene_graph["subject"]["name"]
        actions = list(scene_graph.get("attributes", {}).values())
        objects = [o["name"] for o in scene_graph.get("objects", [])]

        specs: List[Dict[str, Any]] = []

        for action in actions:
            specs.append({
                "dimension": "R_SA",
                "question": TEMPLATES_STRUCTURAL["R_SA"].format(
                    subject=subject, action=action),
                "fields": {"subject": subject, "action": action},
            })

        for obj in objects:
            specs.append({
                "dimension": "R_SO",
                "question": TEMPLATES_STRUCTURAL["R_SO"].format(
                    subject=subject, object=obj),
                "fields": {"subject": subject, "object": obj},
            })

        for action in actions:
            for obj in objects:
                specs.append({
                    "dimension": "R_AO",
                    "question": TEMPLATES_STRUCTURAL["R_AO"].format(
                        action=action, object=obj),
                    "fields": {"action": action, "object": obj},
                })

        return specs

    @staticmethod
    def build_questions_structural(scene_graph: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Build the 3-dimension structural realism questions.

        R(g) = (1/3)(R_SA + R_SO + R_AO)

        Dimensions:
            R_SA: "Can S perform A?"   (subject–action compatibility)
            R_SO: "Can S exist in O?"  (subject–object compatibility)
            R_AO: "Can A occur in O?"  (action–object compatibility)
        """
        return [
            {"dimension": spec["dimension"], "prompt": spec["question"]}
            for spec in RealismScorer.build_question_specs_structural(scene_graph)
        ]

    @staticmethod
    def build_question_specs_from_qa(qa_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build ordered grouped-request specs from pre-generated QA items."""
        specs: List[Dict[str, Any]] = []
        for qa in qa_list:
            fields = {
                key: qa[key]
                for key in ("subject", "attribute", "action", "relation", "object")
                if key in qa and qa[key] not in (None, "")
            }
            if not fields:
                fields = {"question": qa["question"]}
            specs.append({
                "dimension": qa.get("dimension", ""),
                "question": qa["question"],
                "fields": fields,
            })
        return specs

    @staticmethod
    def build_grouped_prompt(specs: List[Dict[str, Any]]) -> str:
        """Build one structured prompt that asks for all scores of one record."""
        lines = [
            f"Score the following {len(specs)} items in order and return exactly {len(specs)} scores.",
        ]
        for idx, spec in enumerate(specs, start=1):
            fields = spec.get("fields", {})
            parts = [f"{idx}. dimension={spec['dimension']}"]
            for key in ("subject", "attribute", "action", "relation", "object", "question"):
                if key in fields:
                    parts.append(f"{key}={json.dumps(fields[key], ensure_ascii=False)}")
            lines.append("; ".join(parts))
        return "\n".join(lines)

    # ----- response parsing -----

    @staticmethod
    def parse_score(response: str) -> Optional[float]:
        """Extract a float in [0, 1] from the LLM response."""
        if response is None:
            return None
        match = re.search(r"(\d+\.?\d*)", response.strip())
        if match:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
        return None

    @staticmethod
    def parse_score_list(response: str, expected_count: int) -> Optional[List[float]]:
        """Extract a list of floats in [0, 1] from a grouped LLM response."""
        if response is None:
            return None

        text = response.strip()
        parsed: Any
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[[^\]]*\]", text, re.S)
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

        if isinstance(parsed, dict):
            parsed = parsed.get("scores")
        if isinstance(parsed, list) and len(parsed) == expected_count:
            scores: List[float] = []
            for item in parsed:
                try:
                    val = float(item)
                except (TypeError, ValueError):
                    return None
                scores.append(max(0.0, min(1.0, val)))
            return scores

        number_matches = re.findall(r"\d+\.?\d*", text)
        if len(number_matches) != expected_count:
            return None
        return [max(0.0, min(1.0, float(item))) for item in number_matches]

    # ----- batch scoring -----

    async def score_records(
        self,
        records: List[Dict[str, Any]],
        max_retries: int = 3,
        mode: str = "default",
        group_questions: bool = False,
        show_progress: bool = True,
    ) -> List[RealismResult]:
        """
        Score a list of scene-graph records.

        Each record is expected to have: id, label, label_name, scene_graph.

        Args:
            records: List of scene-graph records.
            max_retries: Retries per question on parse failure.
            mode: Scoring mode — ``"default"`` (4-dim) or ``"structural"`` (3-dim).
            group_questions: If True, score all questions for one record in a single LLM request.
            show_progress: Whether to display a progress bar on stderr.

        Returns:
            List of ``RealismResult``, one per record.
        """
        if mode not in SCORING_MODES:
            raise ValueError(f"mode must be one of {SCORING_MODES}, got {mode!r}")

        question_builder = (
            self.build_questions_structural if mode == "structural"
            else self.build_questions
        )
        question_spec_builder = (
            self.build_question_specs_structural if mode == "structural"
            else self.build_question_specs
        )

        if group_questions:
            grouped_prompts: List[tuple] = []
            specs_per_record: Dict[int, List[Dict[str, Any]]] = {}
            per_record: Dict[int, List[DimensionScore]] = {i: [] for i in range(len(records))}

            for rec_idx, record in enumerate(records):
                specs = question_spec_builder(record["scene_graph"])
                specs_per_record[rec_idx] = specs
                if specs:
                    grouped_prompts.append((rec_idx, self.build_grouped_prompt(specs)))

            total_calls = len(grouped_prompts)
            logger.info(
                "Scoring %d records with %d grouped LLM calls (mode=%s)",
                len(records), total_calls, mode,
            )

            completed = 0
            t0 = time.time()
            async for rec_idx, response in self.stream_gen.generate_stream_with_index(
                prompts_with_index=grouped_prompts,
                system_prompt=GROUPED_SYSTEM_PROMPT,
                validate_func=None,
            ):
                specs = specs_per_record[rec_idx]
                scores = self.parse_score_list(response, len(specs))
                if scores is None:
                    logger.warning("[rec_idx=%d] Failed to parse grouped scores from: %r", rec_idx, response)
                    scores = [0.5] * len(specs)

                per_record[rec_idx] = [
                    DimensionScore(
                        dimension=spec["dimension"],
                        question=spec["question"],
                        score=score,
                    )
                    for spec, score in zip(specs, scores)
                ]

                completed += 1
                if show_progress and total_calls:
                    elapsed = time.time() - t0
                    eta = elapsed / completed * (total_calls - completed) if completed else 0
                    bar_len = 30
                    filled = int(bar_len * completed / total_calls)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    sys.stderr.write(
                        f"\r  [{bar}] {completed}/{total_calls} grouped calls  "
                        f"[{elapsed:.1f}s < {eta:.1f}s]"
                    )
                    sys.stderr.flush()

            if show_progress and total_calls:
                sys.stderr.write("\n")
                sys.stderr.flush()

            results: List[Optional[RealismResult]] = [None] * len(records)
            for rec_idx, record in enumerate(records):
                if not per_record[rec_idx]:
                    specs = specs_per_record.get(rec_idx, [])
                    per_record[rec_idx] = [
                        DimensionScore(
                            dimension=spec["dimension"],
                            question=spec["question"],
                            score=0.5,
                        )
                        for spec in specs
                    ]

                dim_scores = per_record[rec_idx]
                avg = sum(d.score for d in dim_scores) / len(dim_scores) if dim_scores else 0.0
                results[rec_idx] = RealismResult(
                    id=record["id"],
                    label=record["label"],
                    label_name=record["label_name"],
                    realism_score=avg,
                    dimension_scores=dim_scores,
                    scene_graph=record.get("scene_graph"),
                    caption=record.get("caption", ""),
                )

            return results

        flat_prompts: List[tuple] = []       # (flat_idx, prompt_str)
        index_map: List[tuple] = []          # (record_idx, question_meta)

        for rec_idx, record in enumerate(records):
            questions = question_builder(record["scene_graph"])
            for q in questions:
                flat_idx = len(flat_prompts)
                flat_prompts.append((flat_idx, q["prompt"]))
                index_map.append((rec_idx, q))

        total_calls = len(flat_prompts)
        logger.info(
            "Scoring %d records with %d LLM calls (mode=%s)",
            len(records), total_calls, mode,
        )

        raw_scores: Dict[int, float] = {}
        completed = 0
        t0 = time.time()

        async for flat_idx, response in self.stream_gen.generate_stream_with_index(
            prompts_with_index=flat_prompts,
            system_prompt=SYSTEM_PROMPT,
            validate_func=None,
        ):
            score = self.parse_score(response)
            if score is not None:
                raw_scores[flat_idx] = score
            else:
                logger.warning(f"[idx={flat_idx}] Failed to parse score from: {response!r}")
                raw_scores[flat_idx] = 0.5

            completed += 1
            if show_progress:
                elapsed = time.time() - t0
                eta = elapsed / completed * (total_calls - completed) if completed else 0
                bar_len = 30
                filled = int(bar_len * completed / total_calls)
                bar = "█" * filled + "░" * (bar_len - filled)
                sys.stderr.write(
                    f"\r  [{bar}] {completed}/{total_calls} calls  "
                    f"[{elapsed:.1f}s < {eta:.1f}s]"
                )
                sys.stderr.flush()

        if show_progress:
            sys.stderr.write("\n")
            sys.stderr.flush()

        results: List[Optional[RealismResult]] = [None] * len(records)
        per_record: Dict[int, List[DimensionScore]] = {i: [] for i in range(len(records))}

        for flat_idx, (rec_idx, q_meta) in enumerate(index_map):
            score = raw_scores.get(flat_idx, 0.5)
            per_record[rec_idx].append(DimensionScore(
                dimension=q_meta["dimension"],
                question=q_meta["prompt"],
                score=score,
            ))

        for rec_idx, record in enumerate(records):
            dim_scores = per_record[rec_idx]
            avg = sum(d.score for d in dim_scores) / len(dim_scores) if dim_scores else 0.0
            results[rec_idx] = RealismResult(
                id=record["id"],
                label=record["label"],
                label_name=record["label_name"],
                realism_score=avg,
                dimension_scores=dim_scores,
                scene_graph=record.get("scene_graph"),
                caption=record.get("caption", ""),
            )

        return results

    # ----- file-level convenience -----

    async def score_file(
        self,
        input_path: str,
        output_path: str,
        mode: str = "default",
        batch_size: int = 0,
        group_questions: bool = False,
        show_progress: bool = True,
    ) -> List[RealismResult]:
        """
        Score all records in a scene-graph JSON file and write results.

        Args:
            input_path: Path to input scene-graph JSON.
            output_path: Path to write the scored output JSON.
            mode: ``"default"`` (4-dim) or ``"structural"`` (3-dim R_SA/R_SO/R_AO).
            batch_size: If > 0, process records in batches and flush after each.
                        0 means process all at once.
            group_questions: If True, score all questions for one record in a single LLM request.
            show_progress: Whether to display progress on stderr.

        Returns:
            List of ``RealismResult``.
        """
        in_path, out_path = Path(input_path), Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data["results"]
        dataset_name = data.get("dataset", "")
        logger.info("Loaded %d records from %s (mode=%s)", len(records), in_path, mode)

        desc_map = {
            "default": (
                "Structural realism scores. "
                "R(g) = mean(R_SA, R_SO, R_SRO, R_SCENE). "
                "Each dimension scored by LLM in [0, 1]."
            ),
            "structural": (
                "Structural realism scores (3-dim). "
                "R(g) = mean(R_SA, R_SO, R_AO). "
                "Each dimension scored by LLM in [0, 1]."
            ),
        }

        all_results: List[RealismResult] = []

        if batch_size > 0:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                batch_end = min(i + batch_size, len(records))
                logger.info("Processing batch %d–%d / %d", i, batch_end, len(records))

                batch_results = await self.score_records(
                    batch,
                    mode=mode,
                    group_questions=group_questions,
                    show_progress=show_progress,
                )
                all_results.extend(batch_results)

                self._flush_realism_output(
                    out_path, dataset_name, desc_map.get(mode, ""), all_results,
                )
        else:
            all_results = await self.score_records(
                records,
                mode=mode,
                group_questions=group_questions,
                show_progress=show_progress,
            )
            self._flush_realism_output(
                out_path, dataset_name, desc_map.get(mode, ""), all_results,
            )

        logger.info("Saved %d realism scores to %s", len(all_results), out_path)
        return all_results

    async def score_qa_file(
        self,
        qa_path: str,
        output_path: str,
        flush_every: int = 200,
        group_questions: bool = False,
        show_progress: bool = True,
    ) -> List[RealismResult]:
        """
        Score pre-generated reality-QA questions with streaming output.

        All questions are sent to the LLM at once for maximum parallelism.
        As soon as all questions for a record are answered, the record's
        realism score is computed and appended.  The output file is flushed
        every *flush_every* completed records so partial results are visible.

        Args:
            qa_path: Path to reality-QA JSON (with ``reality_qa`` per record).
            output_path: Path to write scored output JSON.
            flush_every: Flush output file every N completed records.
            group_questions: If True, score all questions for one record in a single LLM request.
            show_progress: Show progress bar on stderr.

        Returns:
            List of ``RealismResult``, one per record.
        """
        qa_p, out_p = Path(qa_path), Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        with open(qa_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        records: List[Dict[str, Any]] = data["results"]
        dataset_name = data.get("dataset", "")
        description = data.get("description", "")
        logger.info("Loaded %d records from %s", len(records), qa_p)

        if group_questions:
            grouped_prompts: List[tuple] = []
            specs_per_record: Dict[int, List[Dict[str, Any]]] = {}
            total_records = len(records)

            for rec_idx, record in enumerate(records):
                specs = self.build_question_specs_from_qa(record.get("reality_qa", []))
                specs_per_record[rec_idx] = specs
                if specs:
                    grouped_prompts.append((rec_idx, self.build_grouped_prompt(specs)))

            total_calls = len(grouped_prompts)
            logger.info(
                "Sending %d grouped prompts for %d records to LLM",
                total_calls, total_records,
            )

            record_done: Dict[int, bool] = {i: False for i in range(total_records)}
            all_results: List[RealismResult] = []
            completed_calls = 0
            completed_records = 0
            last_flush = 0
            t0 = time.time()

            async for rec_idx, response in self.stream_gen.generate_stream_with_index(
                prompts_with_index=grouped_prompts,
                system_prompt=GROUPED_SYSTEM_PROMPT,
                validate_func=None,
            ):
                if record_done[rec_idx]:
                    completed_calls += 1
                    continue

                specs = specs_per_record[rec_idx]
                scores = self.parse_score_list(response, len(specs))
                if scores is None:
                    logger.warning("[rec_idx=%d] Failed to parse grouped scores: %r", rec_idx, response)
                    scores = [0.5] * len(specs)

                dim_scores = [
                    DimensionScore(
                        dimension=spec["dimension"],
                        question=spec["question"],
                        score=score,
                    )
                    for spec, score in zip(specs, scores)
                ]

                rec = records[rec_idx]
                avg = sum(d.score for d in dim_scores) / len(dim_scores) if dim_scores else 0.0
                all_results.append(RealismResult(
                    id=rec["id"],
                    label=rec["label"],
                    label_name=rec["label_name"],
                    realism_score=avg,
                    dimension_scores=dim_scores,
                    scene_graph=rec.get("scene_graph"),
                    caption=rec.get("caption", ""),
                ))
                record_done[rec_idx] = True
                completed_records += 1

                if flush_every > 0 and completed_records - last_flush >= flush_every:
                    self._flush_realism_output(
                        out_p, dataset_name, description, all_results,
                    )
                    last_flush = completed_records

                completed_calls += 1
                if show_progress and total_calls:
                    elapsed = time.time() - t0
                    eta = (
                        elapsed / completed_calls * (total_calls - completed_calls)
                        if completed_calls else 0
                    )
                    bar_len = 30
                    filled = int(bar_len * completed_calls / total_calls)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    sys.stderr.write(
                        f"\r  [{bar}] {completed_calls}/{total_calls} grouped calls  "
                        f"records: {completed_records}/{total_records}  "
                        f"[{elapsed:.1f}s < {eta:.1f}s]"
                    )
                    sys.stderr.flush()

            for rec_idx, done in record_done.items():
                if done:
                    continue
                specs = specs_per_record.get(rec_idx, [])
                dim_scores = [
                    DimensionScore(
                        dimension=spec["dimension"],
                        question=spec["question"],
                        score=0.5,
                    )
                    for spec in specs
                ]
                rec = records[rec_idx]
                avg = sum(d.score for d in dim_scores) / len(dim_scores) if dim_scores else 0.0
                all_results.append(RealismResult(
                    id=rec["id"],
                    label=rec["label"],
                    label_name=rec["label_name"],
                    realism_score=avg,
                    dimension_scores=dim_scores,
                    scene_graph=rec.get("scene_graph"),
                    caption=rec.get("caption", ""),
                ))

            if show_progress and total_calls:
                sys.stderr.write("\n")
                sys.stderr.flush()

            self._flush_realism_output(out_p, dataset_name, description, all_results)
            logger.info("Saved %d realism scores to %s", len(all_results), out_p)
            return all_results

        flat_prompts: List[tuple] = []
        # flat_idx → (rec_idx, qa_item)
        index_map: Dict[int, tuple] = {}
        # rec_idx → total number of questions
        questions_per_record: Dict[int, int] = {}

        for rec_idx, record in enumerate(records):
            qa_list = record.get("reality_qa", [])
            questions_per_record[rec_idx] = len(qa_list)
            for qa in qa_list:
                flat_idx = len(flat_prompts)
                flat_prompts.append((flat_idx, qa["question"]))
                index_map[flat_idx] = (rec_idx, qa)

        total_calls = len(flat_prompts)
        total_records = len(records)
        logger.info(
            "Sending %d questions for %d records to LLM",
            total_calls, total_records,
        )

        per_record_scores: Dict[int, List[DimensionScore]] = {
            i: [] for i in range(total_records)
        }
        record_done: Dict[int, bool] = {i: False for i in range(total_records)}
        seen_flat: set = set()

        all_results: List[RealismResult] = []
        completed_calls = 0
        completed_records = 0
        last_flush = 0
        t0 = time.time()

        async for flat_idx, response in self.stream_gen.generate_stream_with_index(
            prompts_with_index=flat_prompts,
            system_prompt=SYSTEM_PROMPT,
            validate_func=None,
        ):
            if flat_idx in seen_flat:
                completed_calls += 1
                continue
            seen_flat.add(flat_idx)

            score = self.parse_score(response)
            if score is None:
                logger.warning("[idx=%d] Failed to parse: %r", flat_idx, response)
                score = 0.5

            rec_idx, qa = index_map[flat_idx]

            if record_done[rec_idx]:
                completed_calls += 1
                continue

            per_record_scores[rec_idx].append(DimensionScore(
                dimension=qa.get("dimension", ""),
                question=qa["question"],
                score=score,
            ))

            if len(per_record_scores[rec_idx]) >= questions_per_record[rec_idx]:
                record_done[rec_idx] = True
                dim_scores = list(per_record_scores[rec_idx])
                avg = sum(d.score for d in dim_scores) / len(dim_scores)
                rec = records[rec_idx]
                all_results.append(RealismResult(
                    id=rec["id"],
                    label=rec["label"],
                    label_name=rec["label_name"],
                    realism_score=avg,
                    dimension_scores=dim_scores,
                    scene_graph=rec.get("scene_graph"),
                    caption=rec.get("caption", ""),
                ))
                completed_records += 1

                if flush_every > 0 and completed_records - last_flush >= flush_every:
                    self._flush_realism_output(
                        out_p, dataset_name, description, all_results,
                    )
                    last_flush = completed_records

            completed_calls += 1
            if show_progress:
                elapsed = time.time() - t0
                eta = (
                    elapsed / completed_calls * (total_calls - completed_calls)
                    if completed_calls else 0
                )
                bar_len = 30
                filled = int(bar_len * completed_calls / total_calls)
                bar = "█" * filled + "░" * (bar_len - filled)
                sys.stderr.write(
                    f"\r  [{bar}] {completed_calls}/{total_calls} calls  "
                    f"records: {completed_records}/{total_records}  "
                    f"[{elapsed:.1f}s < {eta:.1f}s]"
                )
                sys.stderr.flush()

        if show_progress:
            sys.stderr.write("\n")
            sys.stderr.flush()

        # Final flush
        self._flush_realism_output(out_p, dataset_name, description, all_results)
        logger.info("Saved %d realism scores to %s", len(all_results), out_p)
        return all_results

    @staticmethod
    def _flush_realism_output(
        out_path: Path,
        dataset_name: str,
        description: str,
        results: List[RealismResult],
    ) -> None:
        """Write current realism results to disk."""
        output = {
            "dataset": dataset_name,
            "total": len(results),
            "description": description,
            "results": [r.to_dict() for r in results],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)


__all__ = [
    "RealismScorer", "RealismResult", "DimensionScore",
    "SYSTEM_PROMPT", "TEMPLATES", "TEMPLATES_STRUCTURAL",
]
