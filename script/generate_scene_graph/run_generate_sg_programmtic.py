#!/usr/bin/env python3
"""
Programmatic scene graph generation (no LLM).

Reads co-objects (e.g. coobjects_5.json) and relations (e.g. ralations.json),
samples objects and relations deterministically or randomly, and outputs
scene graphs in the same structure as scene_graphs_compatible_1_2.json.
Scene graph includes attributes.state when a states file is provided.

Inspired by AttrSyn/data_generation/generate_prompts.py (programmatic combinations).

Usage:
    python run_generate_sg_programmtic.py \
        --coobjects_path /path/to/coobjects_5.json \
        --relations_path /path/to/ralations.json \
        --output_path /path/to/scene_graphs.json \
        --samples_per_subject 10 \
        --min_objects 1 \
        --max_objects 2
    # With optional states (adds scene_graph.attributes.state):
    python run_generate_sg_programmtic.py \
        --coobjects_path coobjects_5.json \
        --relations_path ralations.json \
        --states_path states.json \
        --output_path scene_graphs.json
    # Control which components to include (all enabled by default):
    python run_generate_sg_programmtic.py \
        --coobjects_path coobjects_5.json \
        --relations_path ralations.json \
        --output_path scene_graphs.json \
        --no-style \
        --no-relation \
        --no-coobject \
        --no-state
"""

import argparse
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default relations if file has different structure
DEFAULT_RELATIONS = ["on", "near", "above", "in", "next to"]
DEFAULT_STATES = ["perching", "standing", "flying", "resting", "feeding"]
DEFAULT_STYLES = ["portrait photo", "minimalistic photo", "close-up detail photo", "candid photo", "night photo"]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_relations(relations_path: str) -> List[str]:
    data = load_json(relations_path)
    if isinstance(data, list):
        return [str(r) for r in data]
    if isinstance(data, dict) and "relations" in data:
        return [str(r) for r in data["relations"]]
    return DEFAULT_RELATIONS


def load_states_index(states_path: str) -> Dict[str, List[str]]:
    """Build label_name -> list of states."""
    data = load_json(states_path)
    if not isinstance(data, list):
        return {}
    index = {}
    for entry in data:
        label_name = entry.get("label_name")
        states = entry.get("states", [])
        if label_name and states:
            index[label_name] = states
    return index


def load_styles(styles_path: str) -> List[str]:
    """Load styles list from JSON."""
    data = load_json(styles_path)
    if isinstance(data, list):
        return [str(s) for s in data]
    return DEFAULT_STYLES


def load_paintings(paintings_path: str) -> List[str]:
    """Load painting styles list from JSON."""
    data = load_json(paintings_path)
    if isinstance(data, list):
        return [str(s) for s in data]
    return []


def generate_scene_graphs(
    coobjects_path: str,
    relations_path: str,
    output_path: str,
    samples_per_subject: int = 10,
    min_objects: int = 1,
    max_objects: int = 2,
    sampling_category: str = "compatible_non_typical",
    superclass: str = "bird",
    states_path: Optional[str] = None,
    styles_path: Optional[str] = None,
    paintings_path: Optional[str] = None,
    seed: int = 42,
    no_style: bool = False,
    no_relation: bool = False,
    no_coobject: bool = False,
    no_state: bool = False,
    no_painting: bool = False,
) -> Dict[str, Any]:
    """
    Generate scene graphs programmatically (no LLM).

    Args:
        coobjects_path: Path to co-objects JSON (e.g. coobjects_5.json).
        relations_path: Path to relations JSON (e.g. ralations.json), list of relation strings.
        output_path: Where to write scene_graphs JSON.
        samples_per_subject: Number of scene graphs per subject/class.
        min_objects: Minimum co-objects per scene.
        max_objects: Maximum co-objects per scene.
        sampling_category: Key in co-objects to sample from (semantically_associated,
            compatible_non_typical, contextually_contrastive).
        superclass: Word to use after subject in caption (e.g. "bird").
        states_path: Optional path to states JSON (list of {label_name, states: [...]});
            if provided, adds scene_graph.attributes.state.
        styles_path: Optional path to styles JSON (list of style strings);
            if provided, appends random style to caption.
        seed: Random seed for reproducibility.
        no_style: If True, do not add style to caption.
        no_relation: If True, do not add relations to scene graph.
        no_coobject: If True, do not add co-objects to scene graph.
        no_state: If True, do not add state to scene graph attributes.

    Returns:
        The generated dataset dict (dataset, total, results).
    """
    random.seed(seed)

    coobjects = load_json(coobjects_path)
    if not isinstance(coobjects, list):
        raise ValueError("coobjects JSON must be a list of class entries")

    relations = load_relations(relations_path)
    if not relations:
        relations = DEFAULT_RELATIONS

    states_index: Dict[str, List[str]] = {}
    if states_path and Path(states_path).exists():
        states_index = load_states_index(states_path)

    styles: List[str] = DEFAULT_STYLES
    if styles_path and Path(styles_path).exists():
        styles = load_styles(styles_path)

    paintings: List[str] = []
    if paintings_path and Path(paintings_path).exists():
        paintings = load_paintings(paintings_path)

    # Build all combinations per class (like generate_prompts.py)
    dataset_name = "cub_2011_synthetic"
    results: List[Dict[str, Any]] = []
    global_id = 0

    for class_idx, item in enumerate(coobjects):
        label = item.get("label", class_idx)
        label_name = item.get("label_name", "")
        subject = item.get("subject", "")
        if not subject and label_name:
            subject = label_name.replace("_", " ")

        # If no_coobject is True, generate scenes with no co-objects
        if no_coobject:
            all_combos = [[]]  # Empty list means no co-objects
        else:
            object_list = item.get(sampling_category, [])
            if not object_list:
                object_list = (
                    item.get("compatible_non_typical", [])
                    or item.get("semantically_associated", [])
                    or item.get("contextually_contrastive", [])
                )

            # All combinations: choose k objects (k from min_objects to max_objects)
            all_combos = []
            for k in range(min_objects, max_objects + 1):
                if k > len(object_list):
                    continue
                for obj_combo in combinations(object_list, k):
                    # For each combo we can assign any relation to each (subject, obj)
                    # We'll generate one scene per combo with random relations; to get exactly samples_per_subject
                    # we either sample combos or repeat
                    all_combos.append(list(obj_combo))

            if not all_combos:
                continue

        # States for this class (for attributes.state)
        class_states = states_index.get(label_name, DEFAULT_STATES)

        # Sample exactly samples_per_subject scenes for this subject
        if len(all_combos) >= samples_per_subject:
            chosen = random.sample(all_combos, samples_per_subject)
        else:
            chosen = []
            while len(chosen) < samples_per_subject:
                chosen.extend(all_combos)
            chosen = chosen[:samples_per_subject]

        for obj_list in chosen:
            # One relation per object (subject_id=0, object_id=1,2,...)
            rels = random.choices(relations, k=len(obj_list)) if not no_relation else []
            objects_dicts = [{"id": i + 1, "name": obj} for i, obj in enumerate(obj_list)]
            
            # Build relations_dicts only if no_relation is False
            if no_relation:
                relations_dicts = []
            else:
                relations_dicts = [
                    {"subject_id": 0, "object_id": i + 1, "relation": rel}
                    for i, rel in enumerate(rels)
                ]
            
            state_str = random.choice(class_states) if class_states else "perching"

            scene_graph = {
                "id": global_id,
                "subject": {"id": 0, "name": subject},
                "objects": objects_dicts,
                "relations": relations_dicts,
            }
            
            # Add attributes.state only if no_state is False
            if not no_state:
                scene_graph["attributes"] = {"state": state_str}
            else:
                scene_graph["attributes"] = {}

            # Caption: comma-separated, e.g. "a Black footed Albatross bird, flying, in ocean, portrait photo"
            parts = [f"a {subject} {superclass}"]
            
            # Add state only if no_state is False
            if not no_state:
                parts.append(state_str)
            
            # Add objects and relations only if no_coobject and no_relation are False
            if not no_coobject:
                if no_relation:
                    # Only add objects without relations
                    for obj in obj_list:
                        parts.append(obj)
                else:
                    # Add objects with relations
                    for obj, rel in zip(obj_list, rels):
                        parts.append(f"{rel} {obj}")
            
            # Add random style at the end only if no_style is False
            if not no_style:
                style_str = random.choice(styles) if styles else ""
                if style_str:
                    parts.append(style_str)

            # Append painting style (e.g. "oil painting") if paintings provided
            if not no_painting and paintings:
                painting_str = random.choice(paintings)
                if painting_str:
                    parts.append(painting_str)

            caption = ", ".join(parts)

            results.append({
                "id": global_id,
                "label": label,
                "label_name": label_name,
                "scene_graph": scene_graph,
                "caption": caption,
            })
            global_id += 1

    out = {
        "dataset": dataset_name,
        "total": len(results),
        "results": results,
    }
    save_json(out, output_path)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate scene graphs programmatically from co-objects and relations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--coobjects_path",
        type=str,
        required=True,
        help="Path to co-objects JSON (e.g. coobjects_5.json)",
    )
    parser.add_argument(
        "--relations_path",
        type=str,
        required=True,
        help="Path to relations JSON (e.g. ralations.json)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output scene_graphs JSON",
    )
    parser.add_argument(
        "--samples_per_subject",
        type=int,
        default=10,
        help="Number of scene graphs per subject (default: 10)",
    )
    parser.add_argument(
        "--min_objects",
        type=int,
        default=1,
        help="Minimum objects per scene (default: 1)",
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=2,
        help="Maximum objects per scene (default: 2)",
    )
    parser.add_argument(
        "--sampling_category",
        type=str,
        default="compatible_non_typical",
        choices=["semantically_associated", "compatible_non_typical", "contextually_contrastive"],
        help="Which co-object category to sample from (default: compatible_non_typical)",
    )
    parser.add_argument(
        "--superclass",
        type=str,
        default="bird",
        help="Superclass word in caption after subject (default: bird)",
    )
    parser.add_argument(
        "--states_path",
        type=str,
        default=None,
        help="Optional path to states JSON for scene_graph.attributes.state",
    )
    parser.add_argument(
        "--styles_path",
        type=str,
        default=None,
        help="Optional path to styles JSON (list of style strings appended to caption)",
    )
    parser.add_argument(
        "--paintings_path",
        type=str,
        default=None,
        help="Optional path to paintings JSON (list of painting-medium strings appended to caption, e.g. 'oil painting')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no-style",
        action="store_true",
        help="Do not add style to caption",
    )
    parser.add_argument(
        "--no-relation",
        action="store_true",
        help="Do not add relations to scene graph",
    )
    parser.add_argument(
        "--no-coobject",
        action="store_true",
        help="Do not add co-objects to scene graph",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Do not add state to scene graph attributes",
    )
    parser.add_argument(
        "--no-painting",
        action="store_true",
        help="Do not append painting medium to caption",
    )
    args = parser.parse_args()

    generate_scene_graphs(
        coobjects_path=args.coobjects_path,
        relations_path=args.relations_path,
        output_path=args.output_path,
        samples_per_subject=args.samples_per_subject,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        sampling_category=args.sampling_category,
        superclass=args.superclass,
        states_path=args.states_path,
        styles_path=args.styles_path,
        paintings_path=args.paintings_path,
        seed=args.seed,
        no_style=args.no_style,
        no_relation=args.no_relation,
        no_coobject=args.no_coobject,
        no_state=args.no_state,
        no_painting=args.no_painting,
    )
    print(f"Done. Output: {args.output_path}")


if __name__ == "__main__":
    main()
