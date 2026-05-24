#!/usr/bin/env python3
"""
Script to generate distinguishing attributes for each subject.

This script reads co-objects from an input JSON file, queries LLM
to generate distinguishing attributes that help differentiate each subject
from other members of the same superclass.

Usage:
    python run_generate_attribute.py --input_path /path/to/coobjects.json --output_dir /path/to/output_dir --n_attributes 5
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
from src.generators.attribute_generator import AttributeGenerator, AttributeResult, CoObjectItem

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


async def generate_attributes_with_retry(
    stream_gen: StreamGenerator,
    generator: AttributeGenerator,
    item: CoObjectItem,
    index: int,
    superclass: str,
    max_retries: int = 100,
    progress_callback=None
) -> list:
    """
    Generate attributes for a single subject with retry on failure.
    
    Args:
        stream_gen: The StreamGenerator instance
        generator: The AttributeGenerator instance
        item: The co-object item to generate attributes for
        index: The index of the item
        superclass: The broad category for all subjects
        max_retries: Maximum number of retries
        progress_callback: Optional callback function for progress updates
        
    Returns:
        List of attribute dictionaries for this subject
    """
    validator = generator.create_validator()
    prompt = generator.build_prompt(item.subject, superclass)
    system_prompt = generator.get_system_prompt()
    
    for attempt in range(max_retries):
        try:
            if progress_callback:
                progress_callback(f"[{item.subject}] Generating attributes (attempt {attempt + 1})...")
            
            # Generate single response
            async for _, response in stream_gen.generate_stream_with_index(
                prompts_with_index=[(index, prompt)],
                system_prompt=system_prompt,
                validate_func=None
            ):
                if response is None:
                    if progress_callback:
                        progress_callback(f"[{item.subject}] Attempt {attempt + 1} failed: no response")
                    continue
                
                validated_attributes = validator(response)
                
                if validated_attributes is None:
                    if progress_callback:
                        progress_callback(f"[{item.subject}] Attempt {attempt + 1} failed: validation error")
                    continue
                
                # Success!
                if progress_callback:
                    progress_callback(f"[{item.subject}] ✓ Attributes completed ({len(validated_attributes)} attributes)")
                return validated_attributes
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                if progress_callback:
                    progress_callback(f"[{item.subject}] Rate limited, waiting... (attempt {attempt + 1})")
            else:
                if progress_callback:
                    progress_callback(f"[{item.subject}] Error: {error_msg[:50]}")
        
        # Exponential backoff for rate limits
        if attempt < max_retries - 1:
            delay = min(0.5 * (2 ** min(attempt, 5)), 10.0)  # Max 10 seconds
            await asyncio.sleep(delay)
    
    raise RuntimeError(f"Failed to generate attributes for {item.subject} after {max_retries} attempts")


async def generate_single_with_retry(
    stream_gen: StreamGenerator,
    generator: AttributeGenerator,
    item: CoObjectItem,
    index: int,
    superclass: str,
    max_retries: int = 100,
    progress_callback=None
) -> AttributeResult:
    """
    Generate attributes for a single subject.
    
    Args:
        stream_gen: The StreamGenerator instance
        generator: The AttributeGenerator instance
        item: The co-object item to generate attributes for
        index: The index of the item
        superclass: The broad category for all subjects
        max_retries: Maximum number of retries
        progress_callback: Optional callback function for progress updates
        
    Returns:
        AttributeResult for this subject (guaranteed to succeed or raise error)
    """
    if progress_callback:
        progress_callback(f"[Label {index + 1}] Starting: {item.subject}")
    
    # Generate attributes
    attributes = await generate_attributes_with_retry(
        stream_gen=stream_gen,
        generator=generator,
        item=item,
        index=index,
        superclass=superclass,
        max_retries=max_retries,
        progress_callback=progress_callback
    )
    
    if progress_callback:
        progress_callback(f"[Label {index + 1}] ✓ Completed: {item.subject}")
    
    return AttributeResult(
        label=item.label,
        label_name=item.label_name,
        subject=item.subject,
        semantically_associated=item.semantically_associated,
        compatible_non_typical=item.compatible_non_typical,
        contextually_contrastive=item.contextually_contrastive,
        attributes=attributes
    )


async def update_partial_result(
    item: CoObjectItem,
    attributes: list,
    output_path: Path,
    lock: asyncio.Lock,
    partial_results: dict
):
    """
    Update partial result for an item when attributes are generated, and write to file.
    
    Args:
        item: The CoObjectItem
        attributes: The generated attributes
        output_path: Path to the output JSON file
        lock: Async lock for file writing
        partial_results: Dictionary to store partial results (keyed by label)
    """
    try:
        async with lock:
            # Initialize or get existing partial result
            if item.label not in partial_results:
                partial_results[item.label] = {
                    "label": item.label,
                    "label_name": item.label_name,
                    "subject": item.subject,
                    "semantically_associated": item.semantically_associated,
                    "compatible_non_typical": item.compatible_non_typical,
                    "contextually_contrastive": item.contextually_contrastive,
                    "attributes": []
                }
            
            # Update attributes
            partial_results[item.label]["attributes"] = attributes
            
            # Write entire partial results dict to file (maintains JSON array format)
            sorted_results = [partial_results[k] for k in sorted(partial_results.keys())]
            
            # Write to file with explicit flush for real-time streaming
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sorted_results, f, indent=4, ensure_ascii=False)
                f.flush()  # Flush Python buffer
                os.fsync(f.fileno())  # Force write to disk immediately
            
            # Print real-time update to console so user can see file is being updated
            print(f"[FILE UPDATE] ✓ Written {len(sorted_results)} items to {output_path}", flush=True)
            print(f"  Latest: label {item.label} ({item.subject}) - {len(attributes)} attributes", flush=True)
            logger.info(f"✓ Written to file: {len(sorted_results)} total results, label {item.label} ({item.subject}) with {len(attributes)} attributes")
    except Exception as e:
        logger.error(f"Error writing partial result for label {item.label}: {e}", exc_info=True)
        raise


async def generate_attributes(
    generator: AttributeGenerator,
    items: list,
    api_keys: list,
    output_path: Path,
    model_name: str = "deepseek-chat",
    max_concurrent: int = 50,
    superclass: str = "Bird",
    max_retries_per_item: int = 100
) -> list:
    """
    Generate attributes for all items using async streaming.
    Results are written to file in real-time as they complete.
    
    Args:
        generator: The AttributeGenerator instance
        items: List of CoObjectItem objects
        api_keys: List of API keys
        output_path: Path to output file for streaming writes
        model_name: Name of the LLM model to use
        max_concurrent: Maximum concurrent requests per key
        superclass: The broad category for all subjects (e.g., "Bird")
        max_retries_per_item: Maximum retries per individual item
        
    Returns:
        List of AttributeResult objects (guaranteed to have all items)
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
    
    # Initialize empty output file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([], f, indent=4, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    
    logger.info(f"Initialized output file: {output_path}")
    logger.info(f"File exists: {output_path.exists()}, size: {output_path.stat().st_size if output_path.exists() else 0} bytes")
    
    total_tasks = len(items)
    logger.info(f"Starting generation for {len(items)} items...")
    logger.info(f"Results will be written in real-time to: {output_path}")
    
    # Progress tracking with thread-safe counters
    completed_items = {"count": 0}
    lock = asyncio.Lock()
    file_lock = asyncio.Lock()
    partial_results = {}  # Dictionary to store partial results for streaming writes
    all_results = {}  # Dictionary to store complete results
    
    def progress_callback(msg: str):
        """Print progress message in real-time with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}", flush=True)
    
    # Initialize progress bar
    results = []
    pbar = tqdm(
        total=total_tasks, 
        desc="Generating attributes", 
        unit="item",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} items [{elapsed}<{remaining}]",
        position=0,
        leave=True
    )
    
    # Process all items concurrently with individual retry logic
    tasks = []
    for i, item in enumerate(items):
        async def task_wrapper(idx, itm):
            """Wrapper to track progress."""
            def local_progress(msg: str):
                """Local progress callback for messages."""
                progress_callback(msg)
            
            try:
                result = await generate_single_with_retry(
                    stream_gen=stream_gen,
                    generator=generator,
                    item=itm,
                    index=idx,
                    superclass=superclass,
                    max_retries=max_retries_per_item,
                    progress_callback=local_progress
                )
                
                # Write partial result to file immediately
                await update_partial_result(
                    itm, result.attributes, 
                    output_path, file_lock, partial_results
                )
                
                # Store complete result
                async with file_lock:
                    all_results[result.label] = result.to_dict()
                
                async with lock:
                    completed_items["count"] += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        "completed": f"{completed_items['count']}/{len(items)}"
                    })
                
                return result
            except Exception as e:
                async with lock:
                    completed_items["count"] += 1
                    pbar.update(1)
                raise e
        
        tasks.append(task_wrapper(i, item))
    
    try:
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                
                # Update progress bar
                async with lock:
                    current_items = completed_items["count"]
                
                pbar.set_postfix({
                    "completed": f"{current_items}/{len(items)}"
                })
            except Exception as e:
                pbar.close()
                logger.error(f"Error generating for an item: {e}")
                raise e
    finally:
        pbar.close()
    
    # Sort results by label to maintain order
    results.sort(key=lambda r: r.label)
    
    async with lock:
        final_items = completed_items["count"]
    
    logger.info(f"Successfully generated attributes for all {len(results)} items!")
    logger.info(f"Results saved to: {output_path}")
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate distinguishing attributes for subjects using LLM"
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input co-objects JSON file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output JSON file"
    )
    
    parser.add_argument(
        "--n_attributes",
        type=int,
        default=5,
        help="Number of distinguishing attributes to generate per subject (default: 5)"
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
        help="Maximum concurrent requests per API key (default: 5)"
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
        "--max_retries_per_item",
        type=int,
        default=100,
        help="Maximum number of retries per individual item (default: 100)"
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
    output_filename = f"coobjects_with_attributes_{timestamp}.json"
    output_path = output_dir / output_filename
    
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Number of attributes per subject: {args.n_attributes}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Superclass: {args.superclass}")
    
    # Load API keys
    api_keys = load_api_keys(str(config_path))
    
    # Initialize generator
    generator = AttributeGenerator(
        n_attributes=args.n_attributes,
        max_retries=10
    )
    
    # Load co-object items
    items = generator.load_coobjects(str(input_path))
    
    # Run async generation (results are written in real-time during generation)
    results = asyncio.run(
        generate_attributes(
            generator=generator,
            items=items,
            api_keys=api_keys,
            output_path=output_path,
            model_name=args.model_name,
            max_concurrent=args.max_concurrent,
            superclass=args.superclass,
            max_retries_per_item=args.max_retries_per_item
        )
    )
    
    logger.info("Generation completed successfully!")


if __name__ == "__main__":
    main()

