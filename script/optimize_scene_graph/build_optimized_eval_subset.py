#!/usr/bin/env python3
"""
Build an evaluation-ready synthetic subset from an optimized scene-graph JSON.

Input:
1. scene_graphs_optimized_*.json with selection_summary[].selected_ids
2. syn_train_index_*.json with the full synthetic metadata and image paths

Output:
1. A filtered syn_train_index-style JSON
2. A new image directory with per-class folders such as
    001.Black_footed_Albatross/025.png

Example:
    python script/optimize_scene_graph/build_optimized_eval_subset.py \
        --optimized_json data/cub2011_painting/scene_graphs/co_object_5/scene_graphs_optimized_6000_deepseekv4f.json \
        --source_index_json data/cub2011_painting/syn_images/co_object_5/syn_train_index_12000_deepseekv4f.json
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def normalize_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def derive_output_json_name(optimized_json: Path) -> str:
    stem = optimized_json.stem
    if stem.startswith("scene_graphs"):
        return f"{stem.replace('scene_graphs', 'syn_train_index', 1)}.json"
    return f"syn_train_index_{stem}.json"


def derive_output_paths(
    optimized_json: Path,
    source_index_json: Path,
    output_json: str | None,
    output_image_dir: str | None,
) -> tuple[Path, Path]:
    source_parent = source_index_json.parent

    if output_json is None:
        json_path = source_parent / derive_output_json_name(optimized_json)
    else:
        json_path = normalize_path(output_json)

    if output_image_dir is None:
        image_dir = source_parent / optimized_json.stem
    else:
        image_dir = normalize_path(output_image_dir)

    return json_path.resolve(), image_dir.resolve()


def collect_selection_pairs(
    optimized_data: dict[str, Any],
    source_data: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    selection_summary = optimized_data.get("selection_summary")
    if not isinstance(selection_summary, list) or not selection_summary:
        raise ValueError("optimized_json must contain a non-empty selection_summary list")

    source_results = source_data.get("results")
    if not isinstance(source_results, list) or not source_results:
        raise ValueError("source_index_json must contain a non-empty results list")

    source_by_id: dict[int, dict[str, Any]] = {}
    for record in source_results:
        if "id" not in record:
            raise ValueError("each source record must have an id field")
        source_id = int(record["id"])
        if source_id in source_by_id:
            raise ValueError(f"duplicate source id found in source_index_json: {source_id}")
        source_by_id[source_id] = record

    missing_ids: list[int] = []
    duplicate_ids: list[int] = []
    mismatched_labels: list[tuple[int, int, int]] = []
    seen_ids: set[int] = set()
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for class_summary in selection_summary:
        label = int(class_summary["label"])
        selected_ids = class_summary.get("selected_ids", [])
        if not isinstance(selected_ids, list):
            raise ValueError(f"selected_ids must be a list for label {label}")

        for selected_id_raw in selected_ids:
            selected_id = int(selected_id_raw)
            if selected_id in seen_ids:
                duplicate_ids.append(selected_id)
                continue

            source_record = source_by_id.get(selected_id)
            if source_record is None:
                missing_ids.append(selected_id)
                continue

            source_label = int(source_record["label"])
            if source_label != label:
                mismatched_labels.append((selected_id, label, source_label))
                continue

            seen_ids.add(selected_id)
            pairs.append((class_summary, source_record))

    if missing_ids:
        preview = ", ".join(str(item) for item in missing_ids[:10])
        raise ValueError(f"{len(missing_ids)} selected ids not found in source_index_json: {preview}")
    if duplicate_ids:
        preview = ", ".join(str(item) for item in duplicate_ids[:10])
        raise ValueError(f"{len(duplicate_ids)} duplicate selected ids in optimized_json: {preview}")
    if mismatched_labels:
        preview = ", ".join(
            f"id={item_id}: optimized_label={expected}, source_label={actual}"
            for item_id, expected, actual in mismatched_labels[:10]
        )
        raise ValueError(f"{len(mismatched_labels)} selected ids have label mismatches: {preview}")

    expected_total = optimized_data.get("total")
    if expected_total is not None and int(expected_total) != len(pairs):
        raise ValueError(
            "optimized_json total does not match the number of selected ids: "
            f"total={expected_total}, selected={len(pairs)}"
        )

    return pairs


def build_subset(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    output_image_dir: Path,
) -> tuple[list[dict[str, Any]], list[tuple[Path, Path]], dict[int, int]]:
    class_counts: dict[int, int] = defaultdict(int)
    subset: list[dict[str, Any]] = []
    file_ops: list[tuple[Path, Path]] = []
    used_targets: set[Path] = set()

    for _, source_record in pairs:
        label = int(source_record["label"])
        label_name = str(source_record["label_name"])
        old_path = normalize_path(str(source_record["image_path"]))

        class_dir = output_image_dir / f"{label + 1:03d}.{label_name}"
        new_name = old_path.name
        new_path = class_dir / new_name

        if new_path in used_targets:
            raise ValueError(f"duplicate target path generated: {new_path}")

        used_targets.add(new_path)
        class_counts[label] += 1

        new_record = copy.deepcopy(source_record)
        new_record["image_path"] = str(new_path)

        subset.append(new_record)
        file_ops.append((old_path, new_path))

    return subset, file_ops, class_counts


def ensure_output_targets(output_json: Path, output_image_dir: Path, overwrite: bool) -> None:
    if output_json.exists() and not overwrite:
        raise FileExistsError(
            f"output_json already exists: {output_json}. Use --overwrite to replace it."
        )
    if output_image_dir.exists() and any(output_image_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output_image_dir is not empty: {output_image_dir}. Use --overwrite to replace it."
        )
    if overwrite and output_image_dir.exists():
        shutil.rmtree(output_image_dir)


def materialize_files(
    file_ops: list[tuple[Path, Path]],
    copy_mode: str,
    overwrite: bool,
) -> None:
    for old_path, new_path in file_ops:
        if not old_path.exists():
            raise FileNotFoundError(f"source image not found: {old_path}")

        new_path.parent.mkdir(parents=True, exist_ok=True)
        if new_path.exists():
            if not overwrite:
                raise FileExistsError(f"target image already exists: {new_path}")
            if new_path.is_symlink() or new_path.is_file():
                new_path.unlink()

        if copy_mode == "copy":
            shutil.copy2(old_path, new_path)
        elif copy_mode == "symlink":
            new_path.symlink_to(old_path)
        else:
            raise ValueError(f"unsupported copy_mode: {copy_mode}")


def build_output_payload(
    source_data: dict[str, Any],
    subset: list[dict[str, Any]],
    output_image_dir: Path,
) -> dict[str, Any]:
    return {
        "dataset": source_data.get("dataset", "cub_2011_synthetic"),
        "total": len(subset),
        "image_dir": str(output_image_dir),
        "results": subset,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a syn_train_index subset and image folder from optimized scene-graph selections"
    )
    parser.add_argument(
        "--optimized_json",
        type=str,
        required=True,
        help="Path to scene_graphs_optimized_*.json",
    )
    parser.add_argument(
        "--source_index_json",
        type=str,
        required=True,
        help="Path to syn_train_index_*.json that contains the original prompts and image paths",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output syn_train_index JSON path. Default: sibling of source_index_json with an optimized-style name",
    )
    parser.add_argument(
        "--output_image_dir",
        type=str,
        default=None,
        help="Output image root. Default: sibling of source_index_json named after optimized_json stem",
    )
    parser.add_argument(
        "--copy_mode",
        type=str,
        default="copy",
        choices=["copy", "symlink"],
        help="How to materialize images into the new folder while preserving original file names (default: copy)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output JSON and existing target images if they already exist",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate inputs and print the first few planned file operations without writing files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    optimized_json = normalize_path(args.optimized_json)
    source_index_json = normalize_path(args.source_index_json)
    output_json, output_image_dir = derive_output_paths(
        optimized_json=optimized_json,
        source_index_json=source_index_json,
        output_json=args.output_json,
        output_image_dir=args.output_image_dir,
    )

    optimized_data = load_json(optimized_json)
    source_data = load_json(source_index_json)
    pairs = collect_selection_pairs(optimized_data, source_data)
    subset, file_ops, class_counts = build_subset(pairs, output_image_dir)

    if not args.dry_run:
        ensure_output_targets(output_json, output_image_dir, args.overwrite)
        materialize_files(file_ops=file_ops, copy_mode=args.copy_mode, overwrite=args.overwrite)
        output_payload = build_output_payload(source_data, subset, output_image_dir)
        save_json(output_payload, output_json)

    print(f"optimized_json:   {optimized_json}")
    print(f"source_index:     {source_index_json}")
    print(f"output_json:      {output_json}")
    print(f"output_image_dir: {output_image_dir}")
    print(f"selected_total:   {len(subset)}")
    print(f"num_classes:      {len(class_counts)}")
    print(f"copy_mode:        {args.copy_mode}")
    print(f"dry_run:          {args.dry_run}")

    sorted_counts = sorted(class_counts.items())
    if sorted_counts:
        min_count = min(count for _, count in sorted_counts)
        max_count = max(count for _, count in sorted_counts)
        print(f"per_class_count:  min={min_count}, max={max_count}")

    preview_count = min(5, len(file_ops))
    if preview_count:
        print("preview_file_ops:")
        for old_path, new_path in file_ops[:preview_count]:
            print(f"  {old_path} -> {new_path}")

    if not args.dry_run:
        print("subset materialization finished.")


if __name__ == "__main__":
    main()