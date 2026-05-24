#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, List, Any, Tuple


DEFAULT_INPUT = (
    "/mnt/sda/runhaofu/CopulaSyn_v2/results/20260108_200408/"
    "clip_eval_20260108_202559.json"
)
DEFAULT_OUTPUT_DIR = "./output"


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_by_keywords(results: List[Dict[str, Any]], keywords: List[str]) -> Dict[str, Any]:
    for item in results:
        name = str(item.get("name", "")).lower()
        if any(key in name for key in keywords):
            return item
    raise ValueError(f"Result with keywords {keywords} not found.")


def normalize_per_class(pca: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(pca, dict):
        return {str(k): v for k, v in pca.items()}
    if isinstance(pca, list):
        return {str(i): v for i, v in enumerate(pca)}
    raise ValueError("per_class_accuracy must be dict or list.")


def sort_key(class_id: str) -> Tuple[int, str]:
    return (0, int(class_id)) if class_id.isdigit() else (1, class_id)


def build_rows(
    zero_pca: Dict[str, Dict[str, Any]],
    synth_pca: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    common_ids = sorted(set(zero_pca.keys()) & set(synth_pca.keys()), key=sort_key)
    rows: List[Dict[str, Any]] = []
    for class_id in common_ids:
        zero_item = zero_pca[class_id]
        synth_item = synth_pca[class_id]
        name = (
            synth_item.get("name")
            or zero_item.get("name")
            or f"class_{class_id}"
        )
        zero_acc = float(zero_item.get("accuracy", 0.0))
        synth_acc = float(synth_item.get("accuracy", 0.0))
        delta = synth_acc - zero_acc
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "zero_shot_accuracy": zero_acc,
                "synthetic_accuracy": synth_acc,
                "delta": delta,
            }
        )
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]], epsilon: float) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "class_id",
                "class_name",
                "zero_shot_accuracy",
                "synthetic_accuracy",
                "delta",
                "direction",
            ]
        )
        for row in rows:
            delta = row["delta"]
            if delta > epsilon:
                direction = "up"
            elif delta < -epsilon:
                direction = "down"
            else:
                direction = "same"
            writer.writerow(
                [
                    row["class_id"],
                    row["class_name"],
                    f"{row['zero_shot_accuracy']:.6f}",
                    f"{row['synthetic_accuracy']:.6f}",
                    f"{row['delta']:.6f}",
                    direction,
                ]
            )


def write_top_txt(
    path: str,
    top_improved: List[Dict[str, Any]],
    top_decreased: List[Dict[str, Any]],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Top improved (Synthetic - Zero-Shot)\n")
        for idx, row in enumerate(top_improved, start=1):
            f.write(
                f"{idx:>2}. {row['class_name']} (id={row['class_id']}): "
                f"{row['delta']:.6f} "
                f"[zero={row['zero_shot_accuracy']:.6f}, "
                f"syn={row['synthetic_accuracy']:.6f}]\n"
            )
        f.write("\nTop decreased (Synthetic - Zero-Shot)\n")
        for idx, row in enumerate(top_decreased, start=1):
            f.write(
                f"{idx:>2}. {row['class_name']} (id={row['class_id']}): "
                f"{row['delta']:.6f} "
                f"[zero={row['zero_shot_accuracy']:.6f}, "
                f"syn={row['synthetic_accuracy']:.6f}]\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CLIP Zero-Shot vs Synthetic per-class accuracy."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input eval JSON path.")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for outputs."
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
        help="Delta threshold for unchanged classes.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top differences to list.",
    )
    args = parser.parse_args()

    data = load_json(args.input)
    results = data.get("results", [])
    if not results:
        raise ValueError("No results found in input JSON.")

    zero = find_by_keywords(results, ["zero-shot", "zeroshot", "zero shot"])
    synth = find_by_keywords(results, ["synthetic"])

    zero_pca = normalize_per_class(zero.get("per_class_accuracy", {}))
    synth_pca = normalize_per_class(synth.get("per_class_accuracy", {}))

    rows = build_rows(zero_pca, synth_pca)
    if not rows:
        raise ValueError("No overlapping per-class results found.")

    epsilon = args.epsilon
    improved = [r for r in rows if r["delta"] > epsilon]
    decreased = [r for r in rows if r["delta"] < -epsilon]
    unchanged = [r for r in rows if abs(r["delta"]) <= epsilon]

    improved_sorted = sorted(improved, key=lambda r: r["delta"], reverse=True)
    decreased_sorted = sorted(decreased, key=lambda r: r["delta"])
    top_k = max(0, args.top_k)
    top_improved = improved_sorted[:top_k]
    top_decreased = decreased_sorted[:top_k]

    output = {
        "input": os.path.abspath(args.input),
        "zero_shot_name": zero.get("name"),
        "synthetic_name": synth.get("name"),
        "counts": {
            "total": len(rows),
            "improved": len(improved),
            "decreased": len(decreased),
            "unchanged": len(unchanged),
        },
        "top_k": top_k,
        "top_improved": top_improved,
        "top_decreased": top_decreased,
        "improved": improved_sorted,
        "decreased": decreased_sorted,
        "unchanged": unchanged,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    json_path = os.path.join(args.output_dir, f"{base}_zero_vs_synthetic.json")
    csv_path = os.path.join(args.output_dir, f"{base}_zero_vs_synthetic.csv")
    txt_path = os.path.join(args.output_dir, f"{base}_top_diff.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=True, indent=2)
    write_csv(csv_path, rows, epsilon)
    write_top_txt(txt_path, top_improved, top_decreased)

    print("Saved:")
    print(f"- {json_path}")
    print(f"- {csv_path}")
    print(f"- {txt_path}")
    print(
        f"Counts: total={len(rows)} improved={len(improved)} "
        f"decreased={len(decreased)} unchanged={len(unchanged)}"
    )


if __name__ == "__main__":
    main()
