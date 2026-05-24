#!/usr/bin/env python3
"""
从 semantically / compatible / contextually 三个 syn_train_index JSON 文件中，
按「每个类别内」的比例采样，合并成新的 JSON 文件。
合并顺序：先 semantically，再 compatible，再 contextually。

比例含义：例如 0.7 表示从 semantically 文件里每个类别取 70% 的样本。
三个比例可任意（如 0.7, 0.2, 0.1），不要求和为 1。
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def group_by_label(results: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """按 label 分组。"""
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for item in results:
        label = item["label"]
        groups[label].append(item)
    return dict(groups)


def sample_by_ratio(
    items: List[Dict[str, Any]],
    ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """从列表中按比例随机采样。ratio in [0, 1]。"""
    if ratio <= 0:
        return []
    n = len(items)
    k = max(0, min(n, int(round(ratio * n))))
    if k == 0:
        return []
    rng = random.Random(seed)
    return rng.sample(items, k)


def sample_merge(
    compatible_path: str,
    contextually_path: str,
    semantically_path: str,
    ratios: Tuple[float, float, float],
    output_path: str,
    seed: int = 42,
    image_dir_override: Optional[str] = None,
) -> None:
    """
    从三个文件按「每类比例」采样并合并。

    ratios: (semantically_ratio, compatible_ratio, contextually_ratio)
    例如 (0.7, 0.2, 0.1) 表示每类先取 semantically 70%，再 compatible 20%，再 contextually 10%。
    """
    data_c = load_json(compatible_path)
    data_x = load_json(contextually_path)
    data_s = load_json(semantically_path)

    r_c = group_by_label(data_c["results"])
    r_x = group_by_label(data_x["results"])
    r_s = group_by_label(data_s["results"])

    rs, rc, rx = ratios  # semantically, compatible, contextually
    all_labels = sorted(set(r_c.keys()) | set(r_x.keys()) | set(r_s.keys()))

    merged: List[Dict[str, Any]] = []
    new_id = 0

    for label in all_labels:
        # 每个类别用 label 参与 seed，保证可复现且不同类不同随机
        class_seed = seed + label * 10007

        # 顺序：先 semantically，再 compatible，再 contextually
        for items, ratio in [
            (r_s.get(label, []), rs),
            (r_c.get(label, []), rc),
            (r_x.get(label, []), rx),
        ]:
            sampled = sample_by_ratio(items, ratio, class_seed)
            for item in sampled:
                # 深拷贝并重写 id；保留原始 image_path
                entry = json.loads(json.dumps(item))
                entry["id"] = new_id
                if "scene_graph" in entry and isinstance(entry["scene_graph"], dict):
                    entry["scene_graph"]["id"] = new_id
                merged.append(entry)
                new_id += 1

    # 保持与原始 results 相同的键顺序
    out = {
        "dataset": data_c.get("dataset", "cub_2011_synthetic"),
        "total": len(merged),
        "image_dir": image_dir_override
        if image_dir_override is not None
        else data_c.get("image_dir", ""),
        "results": merged,
    }

    # 若未指定 image_dir，在注释里说明各条目的 image_path 为完整路径
    if not out["image_dir"]:
        out["_comment_image_dir"] = "items use full image_path; image_dir empty."

    save_json(out, output_path)
    print(
        f"Written {len(merged)} items to {output_path} "
        f"(ratios semantically={rs}, compatible={rc}, contextually={rx}, seed={seed})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 compatible/contextually/semantically 三个 syn_train_index 按每类比例采样合并"
    )
    parser.add_argument(
        "--compatible",
        type=str,
        default="syn_train_index_compatible.json",
        help="compatible 索引 JSON 路径（默认当前目录下 syn_train_index_compatible.json）",
    )
    parser.add_argument(
        "--contextually",
        type=str,
        default="syn_train_index_contextually.json",
        help="contextually 索引 JSON 路径",
    )
    parser.add_argument(
        "--semantically",
        type=str,
        default="syn_train_index_semantically.json",
        help="semantically 索引 JSON 路径",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=[0.7, 0.2, 0.1],
        metavar=("SEMANTICALLY", "COMPATIBLE", "CONTEXTUALLY"),
        help="每个类别从三个文件采样的比例（顺序：semantically, compatible, contextually），如 0.7 0.2 0.1",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="syn_train_index_merged.json",
        help="输出 JSON 路径",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="可选：输出中 image_dir 字段的值（不设则用 compatible 的 image_dir）",
    )
    args = parser.parse_args()

    r_sem, r_compat, r_context = args.ratios
    if any(r < 0 or r > 1 for r in (r_sem, r_compat, r_context)):
        parser.error("--ratios 三个数均需在 [0, 1] 之间")

    sample_merge(
        compatible_path=args.compatible,
        contextually_path=args.contextually,
        semantically_path=args.semantically,
        ratios=(r_sem, r_compat, r_context),  # semantically, compatible, contextually
        output_path=args.output,
        seed=args.seed,
        image_dir_override=args.image_dir,
    )


if __name__ == "__main__":
    main()
