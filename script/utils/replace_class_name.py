#!/usr/bin/env python3
# 功能: 将输入JSON中的指定类名替换为新类名，并把对应条目的标签改为新值。
# 用法: python replace_class_name.py input.json [-o output.json]
#       [--old-name Pacific_Loon] [--new-name rubbish] [--new-label 200]
#       [--replace-ratio 1.0] [--seed 42]
import argparse
import json
from pathlib import Path
import random


def build_output_path(input_path: Path, new_name: str, new_label: int) -> Path:
    suffix = input_path.suffix
    stem = input_path.stem
    return input_path.with_name(f"{stem}_{new_name}_label{new_label}{suffix}")


def replace_class_entries(
    data: dict,
    old_name: str,
    new_name: str,
    new_label: int,
    replace_ratio: float,
    seed: int,
) -> int:
    if "results" not in data or not isinstance(data["results"], list):
        raise ValueError("Invalid JSON format: missing 'results' list.")

    if replace_ratio < 0 or replace_ratio > 1:
        raise ValueError("replace_ratio must be between 0 and 1.")

    target_indices = []
    for idx, item in enumerate(data["results"]):
        if item.get("label_name") == old_name or item.get("class_name") == old_name:
            target_indices.append(idx)

    total_targets = len(target_indices)
    if total_targets == 0:
        return 0

    n_to_replace = int(total_targets * replace_ratio)
    if n_to_replace == 0:
        return 0

    rng = random.Random(seed)
    selected_indices = set(rng.sample(target_indices, k=n_to_replace))

    changed = 0
    for idx in selected_indices:
        item = data["results"][idx]
        if item.get("label_name") == old_name:
            item["label_name"] = new_name
        if item.get("class_name") == old_name:
            item["class_name"] = new_name
        item["label"] = new_label
        changed += 1

    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace a class name and label in a JSON index file."
    )
    parser.add_argument("input_json", type=Path, help="Input JSON path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: auto-generated next to input)",
    )
    parser.add_argument(
        "--old-name", default="Pacific_Loon", help="Old class name to replace"
    )
    parser.add_argument(
        "--new-name", default="rubbish", help="New class name to use"
    )
    parser.add_argument(
        "--new-label", type=int, default=200, help="New label id to set"
    )
    parser.add_argument(
        "--replace-ratio",
        type=float,
        default=1.0,
        help="Fraction of matching entries to replace (0-1, default: 1.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_json.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_json}")

    output_path = args.output or build_output_path(
        args.input_json, args.new_name, args.new_label
    )

    with args.input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    changed = replace_class_entries(
        data=data,
        old_name=args.old_name,
        new_name=args.new_name,
        new_label=args.new_label,
        replace_ratio=args.replace_ratio,
        seed=args.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Updated {changed} entries.")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
