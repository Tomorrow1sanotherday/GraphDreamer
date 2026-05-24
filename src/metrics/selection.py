"""
Greedy Submodular Selection for Scene Graph Optimization

Selects K scene graphs per class that maximize:

    max_{G* ⊂ G_cand}  λ · D(G*) + (1−λ) · R(g)

    s.t.  |G*| = K

where:
    D(G*) = H(A) + H(O) + H(R)   — structural entropy diversity
    R(g) ∈ [0, 1]                 — per-graph realism score (LLM-based)

The greedy algorithm exploits the monotone sub-modularity of the entropy
function to obtain a constant-factor (1 - 1/e) approximation guarantee.

Algorithm (per class):
    1.  G* ← ∅
    2.  while |G*| < K:
    3.      g* = argmax_{g ∈ G_cand \\ G*} [ λ · ΔD(g) + (1−λ) · R(g) ]
    4.      G* ← G* ∪ {g*}
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SceneGraphTokens:
    """Discrete tokens extracted from a scene graph for diversity computation."""
    actions: Tuple[str, ...]
    objects: Tuple[str, ...]
    relations: Tuple[str, ...]


@dataclass
class SelectionResult:
    """Greedy selection result for one class."""
    label: int
    label_name: str
    num_candidates: int
    num_selected: int
    selected_ids: List[int]
    diversity: float
    avg_realism: float
    objective: float
    selected_records: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "label_name": self.label_name,
            "num_candidates": self.num_candidates,
            "num_selected": self.num_selected,
            "selected_ids": self.selected_ids,
            "diversity": round(self.diversity, 6),
            "avg_realism": round(self.avg_realism, 6),
            "objective": round(self.objective, 6),
        }


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class GreedySceneGraphSelector:
    """
    Greedy submodular maximization for scene graph subset selection.

    For each class (subject category), selects *K* scene graphs from the
    candidate pool that maximize a weighted combination of structural entropy
    diversity and realism score.
    """

    def __init__(
        self,
        budget_per_class: int = 30,
        lambda_weight: float = 0.5,
        log_base: str = "e",
    ) -> None:
        if not 0.0 <= lambda_weight <= 1.0:
            raise ValueError(f"lambda_weight must be in [0, 1], got {lambda_weight}")
        valid_bases = {"e", "2", "10"}
        if log_base not in valid_bases:
            raise ValueError(f"log_base must be one of {valid_bases}, got {log_base}")

        self.budget = budget_per_class
        self.lam = lambda_weight
        self.log_base = log_base

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def select_from_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        flush_every: int = 10,
    ) -> List[SelectionResult]:
        """
        Load candidates, run greedy selection per class, and save.

        Each input record must contain ``realism_score`` and ``scene_graph``.
        Results are streamed to *output_path* every *flush_every* classes so
        that partial progress is visible in the file while the job runs.
        """
        input_path, output_path = Path(input_path), Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        records: List[Dict[str, Any]] = data["results"]
        dataset_name = data.get("dataset", "")
        logger.info("Loaded %d candidate records from %s", len(records), input_path)

        all_results: List[SelectionResult] = []
        all_selected: List[Dict[str, Any]] = []
        summary: List[Dict[str, Any]] = []

        gen = self.select_per_class(records)
        for res in gen:
            all_results.append(res)
            all_selected.extend(res.selected_records)
            summary.append(res.to_dict())

            if len(all_results) % flush_every == 0:
                self._flush_output(
                    output_path, dataset_name,
                    all_selected, summary,
                )

        self._flush_output(output_path, dataset_name, all_selected, summary)

        n_cls = len(all_results)
        avg_d = sum(r.diversity for r in all_results) / n_cls
        avg_r = sum(r.avg_realism for r in all_results) / n_cls
        avg_o = sum(r.objective for r in all_results) / n_cls
        logger.info(
            "Selected %d records across %d classes  |  "
            "avg_D=%.4f  avg_R=%.4f  avg_obj=%.4f",
            len(all_selected), n_cls, avg_d, avg_r, avg_o,
        )
        logger.info("Saved to %s", output_path)
        return all_results

    def select_per_class(
        self,
        records: List[Dict[str, Any]],
    ) -> Generator[SelectionResult, None, None]:
        """Group records by label and yield results one class at a time."""
        by_class: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for rec in records:
            by_class[rec["label"]].append(rec)

        n_classes = len(by_class)
        sorted_labels = sorted(by_class)
        t0 = time.time()

        for idx, label in enumerate(sorted_labels):
            candidates = by_class[label]
            label_name = candidates[0]["label_name"]
            K = min(self.budget, len(candidates))

            selected_indices = self._greedy_select(candidates, K)
            selected_records = [candidates[i] for i in selected_indices]

            div = self._set_diversity(selected_records)
            avg_r = self._set_avg_realism(selected_records)
            obj = self.lam * div + (1.0 - self.lam) * avg_r

            res = SelectionResult(
                label=label,
                label_name=label_name,
                num_candidates=len(candidates),
                num_selected=K,
                selected_ids=[rec["id"] for rec in selected_records],
                diversity=div,
                avg_realism=avg_r,
                objective=obj,
                selected_records=selected_records,
            )

            done = idx + 1
            elapsed = time.time() - t0
            eta = elapsed / done * (n_classes - done) if done else 0
            bar_len = 30
            filled = int(bar_len * done / n_classes)
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stderr.write(
                f"\r  [{bar}] {done}/{n_classes}  "
                f"D={div:.3f} obj={obj:.3f}  "
                f"{label_name:<30s}  "
                f"[{elapsed:.1f}s < {eta:.1f}s]"
            )
            sys.stderr.flush()

            yield res

        sys.stderr.write("\n")
        sys.stderr.flush()

    # ------------------------------------------------------------------ #
    #  Core greedy algorithm                                              #
    # ------------------------------------------------------------------ #

    def _greedy_select(
        self,
        candidates: List[Dict[str, Any]],
        K: int,
    ) -> List[int]:
        """Return indices into *candidates* of the K greedily chosen items."""
        all_tokens = [self._extract_tokens(c["scene_graph"]) for c in candidates]

        selected: List[int] = []
        remaining = set(range(len(candidates)))

        act_ctr: Counter = Counter()
        obj_ctr: Counter = Counter()
        rel_ctr: Counter = Counter()
        cur_div = 0.0

        for _ in range(K):
            best_gain = -math.inf
            best_idx = -1

            for idx in remaining:
                tk = all_tokens[idx]

                delta_d = self._marginal_diversity(
                    tk, act_ctr, obj_ctr, rel_ctr, cur_div,
                )

                r_score = float(candidates[idx].get("realism_score", 0.5))

                gain = self.lam * delta_d + (1.0 - self.lam) * r_score
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            selected.append(best_idx)
            remaining.discard(best_idx)

            tk = all_tokens[best_idx]
            for a in tk.actions:
                act_ctr[a] += 1
            for o in tk.objects:
                obj_ctr[o] += 1
            for r in tk.relations:
                rel_ctr[r] += 1
            cur_div = (
                self._entropy(act_ctr)
                + self._entropy(obj_ctr)
                + self._entropy(rel_ctr)
            )

        return selected

    # ------------------------------------------------------------------ #
    #  Token extraction                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_tokens(scene_graph: Dict[str, Any]) -> SceneGraphTokens:
        """Extract (actions, objects, relations) tokens from a scene graph."""
        actions: List[str] = []
        attrs = scene_graph.get("attributes", {})
        if isinstance(attrs, dict):
            for v in attrs.values():
                tok = str(v).strip() if v is not None else ""
                if tok:
                    actions.append(tok)

        objects: List[str] = []
        for obj in scene_graph.get("objects", []):
            name = obj.get("name", "") if isinstance(obj, dict) else str(obj)
            if name.strip():
                objects.append(name.strip())

        relations: List[str] = []
        for rel in scene_graph.get("relations", []):
            r = rel.get("relation", "") if isinstance(rel, dict) else str(rel)
            if r.strip():
                relations.append(r.strip())

        return SceneGraphTokens(
            actions=tuple(actions),
            objects=tuple(objects),
            relations=tuple(relations),
        )

    # ------------------------------------------------------------------ #
    #  Entropy helpers                                                    #
    # ------------------------------------------------------------------ #

    def _marginal_diversity(
        self,
        tokens: SceneGraphTokens,
        act_ctr: Counter,
        obj_ctr: Counter,
        rel_ctr: Counter,
        cur_div: float,
    ) -> float:
        """Compute ΔD(g) = D(G* ∪ {g}) − D(G*)."""
        tmp_a = Counter(act_ctr)
        tmp_o = Counter(obj_ctr)
        tmp_r = Counter(rel_ctr)

        for a in tokens.actions:
            tmp_a[a] += 1
        for o in tokens.objects:
            tmp_o[o] += 1
        for r in tokens.relations:
            tmp_r[r] += 1

        new_div = self._entropy(tmp_a) + self._entropy(tmp_o) + self._entropy(tmp_r)
        return new_div - cur_div

    def _set_diversity(self, records: List[Dict[str, Any]]) -> float:
        """Compute D(G) = H(A) + H(O) + H(R) for a set of records."""
        act_ctr: Counter = Counter()
        obj_ctr: Counter = Counter()
        rel_ctr: Counter = Counter()

        for rec in records:
            tk = self._extract_tokens(rec["scene_graph"])
            for a in tk.actions:
                act_ctr[a] += 1
            for o in tk.objects:
                obj_ctr[o] += 1
            for r in tk.relations:
                rel_ctr[r] += 1

        return self._entropy(act_ctr) + self._entropy(obj_ctr) + self._entropy(rel_ctr)

    @staticmethod
    def _set_avg_realism(records: List[Dict[str, Any]]) -> float:
        if not records:
            return 0.0
        total = sum(float(rec.get("realism_score", 0.5)) for rec in records)
        return total / len(records)

    def _entropy(self, counter: Counter) -> float:
        """Shannon entropy from a frequency counter."""
        total = sum(counter.values())
        if total == 0:
            return 0.0
        h = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                h -= p * self._log(p)
        return h

    def _log(self, value: float) -> float:
        if self.log_base == "2":
            return math.log2(value)
        if self.log_base == "10":
            return math.log10(value)
        return math.log(value)

    # ------------------------------------------------------------------ #
    #  I/O helpers                                                        #
    # ------------------------------------------------------------------ #

    def _flush_output(
        self,
        output_path: Path,
        dataset_name: str,
        all_selected: List[Dict[str, Any]],
        summary: List[Dict[str, Any]],
    ) -> None:
        """Write current results to disk (overwrites previous snapshot)."""
        output = {
            "dataset": dataset_name,
            "total": len(all_selected),
            "budget_per_class": self.budget,
            "lambda": self.lam,
            "log_base": self.log_base,
            "selection_summary": summary,
            "results": all_selected,
        }
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)



__all__ = [
    "GreedySceneGraphSelector",
    "SelectionResult",
    "SceneGraphTokens",
]
