#!/usr/bin/env python3
"""
Script to generate co-occurring objects for each label class in three categories.

This script reads labels from an input JSON file, queries DeepSeek LLM
to generate objects in three categories:
1. Semantically associated: Objects commonly associated with the subject
2. Compatible but non-typical: Objects that could appear in scene but not typically associated
3. Contextually contrastive: Objects semantically unrelated but could co-exist in scene

Usage:
    python run_generate_coobject.py --input_path /path/to/labels.json --output_dir /path/to/output_dir --n_objects_per_category 20
"""

import os
import sys
import argparse
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import StreamGenerator
from src.generators.coobject_generator import ObjectCooccurrenceGenerator
from src.generators.coobject_generator.generator import CooccurrenceResult, LabelItem, CategoryType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def load_api_keys(config_path: str) -> list:
    """
    Load API keys from the config file.
    
    Args:
        config_path: Path to the key.yaml file
        
    Returns:
        List of API keys
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    keys = config.get("keys", [])
    if not keys:
        raise ValueError("No API keys found in config file")
    
    logger.info(f"Loaded {len(keys)} API keys")
    return keys


async def generate_category_with_retry(
    stream_gen: StreamGenerator,
    generator: ObjectCooccurrenceGenerator,
    label: LabelItem,
    index: int,
    category: CategoryType,
    superclass: str,
    exclude_objects: list,
    max_retries: int = 100,
    progress_callback=None
) -> list:
    """
    Generate co-occurring objects for a single category with retry on failure.
    
    Args:
        stream_gen: The StreamGenerator instance
        generator: The ObjectCooccurrenceGenerator instance
        label: The label item to generate for
        index: The index of the label
        category: The category type to generate
        superclass: The broad category for all subjects
        exclude_objects: List of objects to exclude (from previous categories)
        max_retries: Maximum number of retries
        progress_callback: Optional callback function for progress updates
        
    Returns:
        List of objects for this category
    """
    validator = generator.create_validator(label.display_name, exclude_objects, superclass)
    prompt = generator.build_prompt(label.display_name, superclass, category, exclude_objects)
    system_prompt = generator.get_system_prompt(category)
    
    category_name = category.value.replace("_", " ").title()
    
    for attempt in range(max_retries):
        try:
            if progress_callback:
                progress_callback(f"[{label.display_name}] Generating {category_name} (attempt {attempt + 1})...")
            
            # Generate single response
            async for _, response in stream_gen.generate_stream_with_index(
                prompts_with_index=[(index, prompt)],
                system_prompt=system_prompt,
                validate_func=None
            ):
                if response is None:
                    if progress_callback:
                        progress_callback(f"[{label.display_name}] {category_name} attempt {attempt + 1} failed: no response")
                    continue
                
                validated_objects = validator(response)
                
                if validated_objects is None:
                    if progress_callback:
                        progress_callback(f"[{label.display_name}] {category_name} attempt {attempt + 1} failed: validation error")
                    continue
                
                # Success!
                if progress_callback:
                    progress_callback(f"[{label.display_name}] ✓ {category_name} completed ({len(validated_objects)} objects)")
                return validated_objects
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                if progress_callback:
                    progress_callback(f"[{label.display_name}] {category_name} rate limited, waiting... (attempt {attempt + 1})")
            else:
                if progress_callback:
                    progress_callback(f"[{label.display_name}] {category_name} error: {error_msg[:50]}")
        
        # Exponential backoff for rate limits
        if attempt < max_retries - 1:
            delay = min(0.5 * (2 ** min(attempt, 5)), 10.0)  # Max 10 seconds
            await asyncio.sleep(delay)
    
    raise RuntimeError(f"Failed to generate {category.value} for {label.display_name} after {max_retries} attempts")


async def generate_single_with_retry(
    stream_gen: StreamGenerator,
    generator: ObjectCooccurrenceGenerator,
    label: LabelItem,
    index: int,
    superclass: str,
    max_retries: int = 100,
    progress_callback=None,
    category_complete_callback=None
) -> CooccurrenceResult:
    """
    Generate co-occurring objects for a single label in all three categories.
    Each category is generated separately with its own prompt to ensure quality.
    
    Args:
        stream_gen: The StreamGenerator instance
        generator: The ObjectCooccurrenceGenerator instance
        label: The label item to generate for
        index: The index of the label
        superclass: The broad category for all subjects
        max_retries: Maximum number of retries per category
        progress_callback: Optional callback function for progress updates
        category_complete_callback: Optional async callback when a category completes (receives category, objects)
        
    Returns:
        CooccurrenceResult for this label (guaranteed to succeed or raise error)
    """
    # Track all generated objects to avoid duplicates across categories
    all_objects = []
    results = {}
    
    if progress_callback:
        progress_callback(f"[Label {index + 1}] Starting: {label.display_name}")
    
    # Generate each category sequentially to accumulate exclusion list
    for cat_idx, category in enumerate(generator.ALL_CATEGORIES, 1):
        objects = await generate_category_with_retry(
            stream_gen=stream_gen,
            generator=generator,
            label=label,
            index=index,
            category=category,
            superclass=superclass,
            exclude_objects=all_objects,
            max_retries=max_retries,
            progress_callback=progress_callback
        )
        results[category.value] = objects
        all_objects.extend(objects)
        
        # Notify that a category is complete with category info and objects
        if category_complete_callback:
            if not objects or len(objects) == 0:
                logger.warning(f"Category {category.value} completed for {label.display_name} but objects list is empty!")
            await category_complete_callback(label, category, objects)
        
        if progress_callback:
            progress_callback(f"[Label {index + 1}] Category {cat_idx}/3 completed for {label.display_name}")
    
    if progress_callback:
        progress_callback(f"[Label {index + 1}] ✓ Completed: {label.display_name}")
    
    return CooccurrenceResult(
        label=label.label,
        label_name=label.label_name,
        subject=label.display_name,
        semantically_associated=results[CategoryType.SEMANTICALLY_ASSOCIATED.value],
        compatible_non_typical=results[CategoryType.COMPATIBLE_NON_TYPICAL.value],
        contextually_contrastive=results[CategoryType.CONTEXTUALLY_CONTRASTIVE.value]
    )


async def update_partial_result(
    label: LabelItem,
    category: CategoryType,
    objects: list,
    output_path: Path,
    lock: asyncio.Lock,
    partial_results: dict
):
    """
    Update partial result for a label when a category completes, and write to file.
    
    Args:
        label: The LabelItem
        category: The completed category
        objects: The generated objects for this category
        output_path: Path to the output JSON file
        lock: Async lock for file writing
        partial_results: Dictionary to store partial results (keyed by label)
    """
    try:
        async with lock:
            # Initialize or get existing partial result
            if label.label not in partial_results:
                partial_results[label.label] = {
                    "label": label.label,
                    "label_name": label.label_name,
                    "subject": label.display_name,
                    "semantically_associated": [],
                    "compatible_non_typical": [],
                    "contextually_contrastive": []
                }
            
            # Update the specific category
            partial_results[label.label][category.value] = objects
            
            # Write entire partial results dict to file (maintains JSON array format)
            sorted_results = [partial_results[k] for k in sorted(partial_results.keys())]
            
            # Write to file with explicit flush
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sorted_results, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            logger.info(f"✓ Written to file: {len(sorted_results)} total results, label {label.label} ({label.display_name}) category {category.value} with {len(objects)} objects")
    except Exception as e:
        logger.error(f"Error writing partial result for label {label.label}, category {category.value}: {e}", exc_info=True)
        raise


async def write_result_to_file(result: CooccurrenceResult, output_path: Path, lock: asyncio.Lock, all_results: dict):
    """
    Write a single complete result to file in real-time (streaming output).
    
    Args:
        result: The CooccurrenceResult to write
        output_path: Path to the output JSON file
        lock: Async lock for file writing
        all_results: Dictionary to store all results (keyed by label)
    """
    async with lock:
        # Store result in dictionary
        all_results[result.label] = result.to_dict()
        
        # Write entire results dict to file (maintains JSON array format)
        sorted_results = [all_results[k] for k in sorted(all_results.keys())]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sorted_results, f, indent=4, ensure_ascii=False)


async def generate_coobjects(
    generator: ObjectCooccurrenceGenerator,
    labels: list,
    api_keys: list,
    output_path: Path,
    model_name: str = "deepseek-chat",
    max_concurrent: int = 50,
    superclass: str = "Bird",
    max_retries_per_label: int = 100
) -> list:
    """
    Generate co-occurring objects for all labels using async streaming.
    Each label generates three categories separately for better quality.
    Each category is retried immediately upon failure until success.
    Results are written to file in real-time as they complete.
    
    Args:
        generator: The ObjectCooccurrenceGenerator instance
        labels: List of LabelItem objects
        api_keys: List of API keys
        output_path: Path to output file for streaming writes
        model_name: Name of the LLM model to use
        max_concurrent: Maximum concurrent requests per key
        superclass: The broad category for all subjects (e.g., "Bird")
        max_retries_per_label: Maximum retries per individual label per category
        
    Returns:
        List of CooccurrenceResult objects (guaranteed to have all labels)
    """
    # Initialize the stream generator
    stream_gen = StreamGenerator(
        model_name=model_name,
        api_keys=api_keys,
        max_concurrent_per_key=max_concurrent,
        max_retries=5,
        rational=False
    )
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize empty output file with a comment/status
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([], f, indent=4, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    
    logger.info(f"Initialized output file: {output_path}")
    logger.info(f"File exists: {output_path.exists()}, size: {output_path.stat().st_size if output_path.exists() else 0} bytes")
    
    total_tasks = len(labels) * 3  # 3 categories per label
    logger.info(f"Starting generation for {len(labels)} labels (3 categories each = {total_tasks} total tasks)...")
    logger.info(f"Results will be written in real-time to: {output_path}")
    
    # Progress tracking with thread-safe counters
    completed_labels = {"count": 0}
    completed_categories = {"count": 0}
    lock = asyncio.Lock()
    file_lock = asyncio.Lock()
    partial_results = {}  # Dictionary to store partial results for streaming writes (updated per category)
    all_results = {}  # Dictionary to store complete results
    
    def progress_callback(msg: str):
        """Print progress message in real-time with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Use logger for better formatting, but print immediately
        print(f"[{timestamp}] {msg}", flush=True)
    
    # Initialize progress bar before creating tasks (so callbacks can access it)
    results = []
    pbar = tqdm(
        total=total_tasks, 
        desc="Categories", 
        unit="category",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} categories [{elapsed}<{remaining}]",
        position=0,
        leave=True
    )
    
    # Process all labels concurrently with individual retry logic
    tasks = []
    for i, label in enumerate(labels):
        async def task_wrapper(idx, lbl):
            """Wrapper to track progress."""
            def local_progress(msg: str):
                """Local progress callback for messages."""
                progress_callback(msg)
            
            async def on_category_complete(label_item, category, objects):
                """Callback when a category completes - writes to file immediately and updates progress bar."""
                try:
                    logger.info(f"on_category_complete called: {label_item.display_name}, category {category.value}, {len(objects)} objects")
                    
                    # Validate objects
                    if not objects:
                        logger.warning(f"Empty objects list for {label_item.display_name}, category {category.value}")
                    
                    # Update category counter
                    async with lock:
                        completed_categories["count"] += 1
                        current_categories = completed_categories["count"]
                        current_labels = completed_labels["count"]
                    
                    # Write partial result to file immediately (streaming output per category)
                    await update_partial_result(
                        label_item, category, objects, 
                        output_path, file_lock, partial_results
                    )
                    
                    # Update progress bar per category (not per label)
                    pbar.update(1)
                    pbar.set_postfix({
                        "labels": f"{current_labels}/{len(labels)}",
                        "categories": f"{current_categories}/{total_tasks}"
                    })
                except Exception as e:
                    logger.error(f"Error in on_category_complete for {label_item.display_name}, category {category.value}: {e}", exc_info=True)
                    raise
            
            try:
                result = await generate_single_with_retry(
                    stream_gen=stream_gen,
                    generator=generator,
                    label=lbl,
                    index=idx,
                    superclass=superclass,
                    max_retries=max_retries_per_label,
                    progress_callback=local_progress,
                    category_complete_callback=on_category_complete
                )
                
                # Store complete result (file already updated per category)
                async with file_lock:
                    all_results[result.label] = result.to_dict()
                
                async with lock:
                    completed_labels["count"] += 1
                
                return result
            except Exception as e:
                async with lock:
                    completed_labels["count"] += 1
                raise e
        
        tasks.append(task_wrapper(i, label))
    
    try:
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                
                # Update progress bar with current counts (progress bar updates per category in callback)
                async with lock:
                    current_labels = completed_labels["count"]
                    current_categories = completed_categories["count"]
                
                # Don't update progress bar here - it's updated per category in on_category_complete
                pbar.set_postfix({
                    "labels": f"{current_labels}/{len(labels)}",
                    "categories": f"{current_categories}/{total_tasks}"
                })
            except Exception as e:
                pbar.close()
                logger.error(f"Error generating for a label: {e}")
                raise e
    finally:
        pbar.close()
    
    # Sort results by label to maintain order
    results.sort(key=lambda r: r.label)
    
    async with lock:
        final_categories = completed_categories["count"]
    
    logger.info(f"Successfully generated all {len(results)} labels ({final_categories} categories total)!")
    logger.info(f"Results saved to: {output_path}")
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate co-occurring objects for label classes using LLM"
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input labels JSON file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output JSON file"
    )
    
    parser.add_argument(
        "--n_objects_per_category",
        type=int,
        default=20,
        help="Number of co-occurring objects to generate per category per subject (default: 20)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-v3",
        help="Name of the LLM model to use (default: deepseek-v3)"
    )
    
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=5,
        help="Maximum concurrent requests per API key (default: 50)"
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the key.yaml config file (default: config/key.yaml)"
    )
    
    parser.add_argument(
        "--superclass",
        type=str,
        default="Bird",
        help="The broad category for all subjects (default: Bird for CUB-200)"
    )
    
    parser.add_argument(
        "--max_retries_per_label",
        type=int,
        default=100,
        help="Maximum number of retries per individual label (default: 100)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Resolve paths
    if args.config_path is None:
        config_path = PROJECT_ROOT / "config" / "key.yaml"
    else:
        config_path = Path(args.config_path)
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"coobjects_{timestamp}.json"
    output_path = output_dir / output_filename
    
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Number of objects per category: {args.n_objects_per_category}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Superclass: {args.superclass}")
    
    # Load API keys
    api_keys = load_api_keys(str(config_path))
    
    # Load labels first so we can pass all names to the generator
    with open(input_path, "r", encoding="utf-8") as f:
        raw_labels = json.load(f)
    all_label_names = [item["label_name"] for item in raw_labels]

    # Initialize generator with dataset class names for same-class filtering
    generator = ObjectCooccurrenceGenerator(
        n_objects_per_category=args.n_objects_per_category,
        max_retries=10,
        all_label_names=all_label_names,
    )

    # Load labels
    labels = generator.load_labels(str(input_path))
    
    # Run async generation (results are written in real-time during generation)
    results = asyncio.run(
        generate_coobjects(
            generator=generator,
            labels=labels,
            api_keys=api_keys,
            output_path=output_path,
            model_name=args.model_name,
            max_concurrent=args.max_concurrent,
            superclass=args.superclass,
            max_retries_per_label=args.max_retries_per_label
        )
    )
    
    logger.info("Generation completed successfully!")
    

if __name__ == "__main__":
    main()

