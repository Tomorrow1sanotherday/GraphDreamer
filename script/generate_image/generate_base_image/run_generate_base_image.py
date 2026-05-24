#!/usr/bin/env python3
"""
Generate images from a base_prompts.json file (produced by AttrSyn/data_generation/
generate_base_prompts.py), reusing the existing multi-GPU image generation pipeline.

This is the base-prompt counterpart of run_generate_image.py: instead of taking a
rich scene_graphs JSON, it takes a flat list of prompts in the AttrSyn base-prompt
schema and adapts it on the fly. The output layout matches the scene-graph pipeline
so downstream evaluation is unchanged:

    <output_dir>/{label+1:03d}.{label_name}/{local_index:03d}.{ext}
    <output_json>  -- syn_train_index style JSON with caption + image_path per entry

Usage:
    python run_generate_image_from_base.py \
        --base_prompts_path data/cub2011/base_prompts.json \
        --output_dir data/cub2011/syn_images/co_object_5/base_prompts_6000 \
        --output_json data/cub2011/syn_images/co_object_5/syn_train_index_base_6000.json \
        --dataset_name cub_2011_synthetic \
        --gpu_ids 0,1,2,3,4,5,6,7
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generators.image_generator import (
    GeneratorConfig,
    generate_images_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def base_prompts_to_scene_graph_payload(
    base_prompts_path: Path,
    dataset_name: str,
) -> Dict:
    """
    Convert a base_prompts.json into the scene-graph JSON schema expected by
    ImageGenerationManager.load_scene_graphs.

    Each base prompt becomes one "result" with a minimal scene_graph stub
    (subject = class_name, no objects/relations). Task IDs are assigned in
    enumeration order so that, when grouped by label, local indices within a
    class match the original base_prompts ordering.
    """
    with open(base_prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts: List[Dict] = data.get("prompts", [])
    if not prompts:
        raise ValueError(f"No prompts found in {base_prompts_path}")

    # The image_generator uses `label_name` to build the class folder
    # ("{label+1:03d}.{label_name}"). The base prompts include `label_name`
    # straight from labels.json, which matches the convention used by the
    # scene-graph pipeline (e.g. "001.Black_footed_Albatross").
    results = []
    for task_id, p in enumerate(prompts):
        if "label_name" in p:
            label_name = p["label_name"]
        else:
            # Fall back to folder_name ("001.Foo_Bar" -> "Foo_Bar") for
            # legacy base_prompts files without a label_name field.
            folder = p["folder_name"]
            label_name = folder.split(".", 1)[1] if "." in folder else folder

        subject_name = p.get("class_name", label_name.replace("_", " "))

        results.append({
            "id": task_id,
            "label": p["class_id"],
            "label_name": label_name,
            "scene_graph": {
                "id": task_id,
                "subject": {"id": 0, "name": subject_name},
                "objects": [],
                "relations": [],
            },
            "caption": p["prompt"],
        })

    return {
        "dataset": dataset_name,
        "total": len(results),
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images from base_prompts.json using the existing multi-GPU pipeline."
    )
    parser.add_argument("--base_prompts_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--image_height", type=int, default=1024)
    parser.add_argument(
        "--image_format", type=str, default="png", choices=["jpg", "png"]
    )
    parser.add_argument("--filename_prefix", type=str, default="syn_image")
    parser.add_argument("--dataset_name", type=str, default="synthetic")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--no_safetensors", action="store_true")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU ids, e.g. '0,1,2,3'. If omitted, derived from --device.",
    )
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="Keep the intermediate scene_graphs JSON written next to --output_json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_prompts_path = Path(args.base_prompts_path)
    if not base_prompts_path.exists():
        logger.error(f"base_prompts file not found: {base_prompts_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_json = Path(args.output_json)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Convert base prompts to scene-graph payload and persist it so the
    # existing loader (which reads from disk) can consume it.
    payload = base_prompts_to_scene_graph_payload(
        base_prompts_path=base_prompts_path,
        dataset_name=args.dataset_name,
    )

    intermediate_path = output_json.with_name(
        output_json.stem + "_source_scene_graphs.json"
    )
    with open(intermediate_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(
        f"Wrote intermediate scene_graphs payload "
        f"({payload['total']} items) to {intermediate_path}"
    )

    # Parse GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
        if not gpu_ids:
            gpu_ids = None
    if gpu_ids is None and args.device.startswith("cuda:"):
        try:
            gpu_ids = [int(args.device.split(":")[1])]
        except (ValueError, IndexError):
            gpu_ids = [0]
    elif gpu_ids is None and args.device == "cuda":
        gpu_ids = [0]

    config = GeneratorConfig(
        model_name=args.model,
        device=args.device,
        torch_dtype=args.dtype,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        image_width=args.image_width,
        image_height=args.image_height,
        image_format=args.image_format,
        filename_prefix=args.filename_prefix,
        use_safetensors=not args.no_safetensors,
        enable_xformers=args.enable_xformers,
    )

    print("=" * 70)
    print("Image Generation from Base Prompts")
    print("=" * 70)
    print(f"Base prompts:      {base_prompts_path}")
    print(f"Intermediate SG:   {intermediate_path}")
    print(f"Output Directory:  {output_dir}")
    print(f"Output JSON:       {output_json}")
    print(f"Model:             {args.model}")
    print(f"Device:            {args.device}")
    if gpu_ids:
        print(f"GPU IDs:           {gpu_ids} ({len(gpu_ids)} GPU(s))")
    print(f"Dtype:             {args.dtype}")
    print(f"Seed:              {args.seed}")
    print(
        f"Guidance Scale:    "
        f"{args.guidance_scale if args.guidance_scale is not None else '(using model default)'}"
    )
    print(f"Inference Steps:   {args.num_inference_steps}")
    print(f"Image Size:        {args.image_width}x{args.image_height}")
    print(f"Image Format:      {args.image_format}")
    print(f"Resume Mode:       {not args.no_resume}")
    print("=" * 70)

    try:
        generated, skipped = generate_images_batch(
            input_json_path=str(intermediate_path),
            output_dir=str(output_dir),
            output_json_path=str(output_json),
            config=config,
            dataset_name=args.dataset_name,
            resume=not args.no_resume,
            gpu_ids=gpu_ids,
        )

        print("\n" + "=" * 70)
        print("Generation Summary")
        print("=" * 70)
        print(f"Newly Generated:   {generated}")
        print(f"Skipped (cached):  {skipped}")
        print(f"Total:             {generated + skipped}")
        print("=" * 70)
        print("\nDone!")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if not args.keep_intermediate:
            try:
                intermediate_path.unlink()
                logger.info(f"Removed intermediate scene_graphs file: {intermediate_path}")
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
