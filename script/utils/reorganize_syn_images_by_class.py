#!/usr/bin/env python3
"""
将测评数据的图片从「单文件夹平铺」重组为「每类一个子文件夹」结构，
与 sdxl-baseprompt-30per 类似：001.Black_footed_Albatross、002.Laysan_Albatross 等。

用法:
  python reorganize_syn_images_by_class.py --syn_index path/to/syn_train_index_semantically_1_1.json
  python reorganize_syn_images_by_class.py --syn_index path/to/syn_train_index.json --output_dir path/to/syn_images_by_class

会生成:
  - 新目录: 每类一个子文件夹，命名为 {label+1:03d}.{label_name}，如 001.Black_footed_Albatross
  - 新 JSON: 与输入同目录（或由 --output_json 指定），image_path 和 image_dir 已更新
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize synthetic images into per-class folders (sdxl-baseprompt-30per style)"
    )
    parser.add_argument(
        "--syn_index",
        type=str,
        required=True,
        help="Path to syn_train_index JSON (e.g. syn_train_index_semantically_1_1.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory for new per-class folders. Default: same parent as current image_dir, name = current_image_dir_name + '_by_class'",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path for new index JSON. Default: same dir as --syn_index, filename with _by_class before .json",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move images instead of copying (default: copy, to keep original folder)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned folders and paths, do not create files",
    )
    args = parser.parse_args()

    syn_index_path = Path(args.syn_index)
    if not syn_index_path.exists():
        raise FileNotFoundError(f"syn_index not found: {syn_index_path}")

    with open(syn_index_path, "r") as f:
        data = json.load(f)

    image_dir = Path(data["image_dir"])
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir from index does not exist: {image_dir}")

    # 输出根目录：默认与当前 image_dir 同父目录，名为 image_dir.name + '_by_class'
    if args.output_dir is not None:
        output_root = Path(args.output_dir)
    else:
        output_root = image_dir.parent / f"{image_dir.name}_by_class"

    # 新 JSON 路径
    if args.output_json is not None:
        output_json_path = Path(args.output_json)
    else:
        output_json_path = syn_index_path.parent / f"{syn_index_path.stem}_by_class.json"

    results = data["results"]
    # 按类分组，保证每类内顺序不变（用于生成 000.png, 001.png, ...）
    by_class = defaultdict(list)
    for r in results:
        by_class[r["label"]].append(r)

    new_results = []
    for label in sorted(by_class.keys()):
        samples = by_class[label]
        label_name = samples[0]["label_name"]
        folder_name = f"{label + 1:03d}.{label_name}"
        class_dir = output_root / folder_name

        if args.dry_run:
            print(f"Would create: {class_dir}")
        else:
            class_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(samples):
            old_path = Path(sample["image_path"])
            ext = old_path.suffix
            new_filename = f"{i:03d}{ext}"
            new_path = class_dir / new_filename

            if args.dry_run:
                print(f"  {old_path.name} -> {new_path}")
            else:
                if not old_path.exists():
                    raise FileNotFoundError(f"Image not found: {old_path}")
                if args.move:
                    shutil.move(str(old_path), str(new_path))
                else:
                    shutil.copy2(old_path, new_path)

            new_sample = {**sample, "image_path": str(new_path)}
            new_results.append(new_sample)

    new_data = {
        "dataset": data.get("dataset", "cub_2011_synthetic"),
        "total": data.get("total", len(new_results)),
        "image_dir": str(output_root),
        "results": new_results,
    }

    if args.dry_run:
        print(f"Would write JSON: {output_json_path}")
        return

    with open(output_json_path, "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"Done. Class folders: {output_root}")
    print(f"New index: {output_json_path}")
    print(f"Total samples: {len(new_results)}")


if __name__ == "__main__":
    main()
