#!/usr/bin/env python3
"""
Script to generate images from scene graphs.

This script reads scene graphs from an input JSON file, generates images
using a diffusion model, and saves results with streaming output.

Usage:
    python run_generate_image.py \
        --input_path /path/to/scene_graphs.json \
        --output_dir /path/to/images \
        --output_json /path/to/output.json
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generators.image_generator import (
    GeneratorConfig,
    generate_images_batch
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate images from scene graphs using diffusion models"
    )
    
    # Input/Output paths
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input scene graphs JSON file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated images"
    )
    
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to save output JSON file with image paths"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Model name or path (default: stabilityai/stable-diffusion-xl-base-1.0)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu, default: cuda)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Torch dtype (default: float16)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Guidance scale for generation (optional, uses model default if not specified)"
    )
    
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)"
    )
    
    parser.add_argument(
        "--image_width",
        type=int,
        default=1024,
        help="Generated image width (default: 1024)"
    )
    
    parser.add_argument(
        "--image_height",
        type=int,
        default=1024,
        help="Generated image height (default: 1024)"
    )
    
    parser.add_argument(
        "--image_format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Image format (default: jpg)"
    )
    
    parser.add_argument(
        "--filename_prefix",
        type=str,
        default="syn_image",
        help="Prefix for generated image filenames (default: syn_image)"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="synthetic",
        help="Name of the dataset (default: synthetic)"
    )
    
    # Resume configuration
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resume mode (regenerate all images)"
    )
    
    # Memory optimizations
    parser.add_argument(
        "--enable_xformers",
        action="store_true",
        help="Enable xformers memory efficient attention"
    )
    
    parser.add_argument(
        "--no_safetensors",
        action="store_true",
        help="Disable safetensors loading"
    )
    
    # Multi-GPU configuration
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="GPU ID(s) to use. Can be a single GPU (e.g., '2') or comma-separated list (e.g., '0,1,2'). "
             "If not specified, automatically extracts GPU ID from --device parameter (e.g., cuda:2 -> GPU 2)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate input file exists
    input_path = Path(args.input_path)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Parse GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        try:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
            if not gpu_ids:
                logger.warning("No valid GPU IDs provided, will extract from --device parameter")
                gpu_ids = None
        except ValueError as e:
            logger.error(f"Invalid GPU IDs format: {args.gpu_ids}. Error: {e}")
            sys.exit(1)
    
    # If gpu_ids not specified, try to extract from device parameter
    if gpu_ids is None and args.device.startswith("cuda:"):
        try:
            gpu_id = int(args.device.split(":")[1])
            gpu_ids = [gpu_id]
            logger.info(f"Extracted GPU ID {gpu_id} from device parameter: {args.device}")
        except (ValueError, IndexError):
            # If device is just "cuda", default to GPU 0
            if args.device == "cuda":
                gpu_ids = [0]
                logger.info("Device is 'cuda', defaulting to GPU 0")
    
    # Create configuration
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
        enable_xformers=args.enable_xformers
    )
    
    # Print configuration
    print("=" * 70)
    print("Image Generation from Scene Graphs")
    print("=" * 70)
    print(f"Input JSON:        {args.input_path}")
    print(f"Output Directory:  {args.output_dir}")
    print(f"Output JSON:       {args.output_json}")
    print(f"Model:             {args.model}")
    print(f"Device:            {args.device}")
    if gpu_ids:
        if len(gpu_ids) == 1:
            print(f"GPU ID:            {gpu_ids[0]}")
        else:
            print(f"GPU IDs:           {gpu_ids} ({len(gpu_ids)} GPU(s))")
    print(f"Dtype:             {args.dtype}")
    print(f"Seed:              {args.seed}")
    print(f"Guidance Scale:    {args.guidance_scale if args.guidance_scale is not None else '(using model default)'}")
    print(f"Inference Steps:   {args.num_inference_steps}")
    print(f"Image Size:        {args.image_width}x{args.image_height}")
    print(f"Image Format:      {args.image_format}")
    print(f"Resume Mode:       {not args.no_resume}")
    print("=" * 70)
    
    # Run generation
    try:
        generated, skipped = generate_images_batch(
            input_json_path=args.input_path,
            output_dir=args.output_dir,
            output_json_path=args.output_json,
            config=config,
            dataset_name=args.dataset_name,
            resume=not args.no_resume,
            gpu_ids=gpu_ids
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


if __name__ == "__main__":
    main()

