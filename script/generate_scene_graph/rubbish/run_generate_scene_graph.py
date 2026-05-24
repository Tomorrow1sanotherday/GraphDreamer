#!/usr/bin/env python3
"""
Script to generate scene graphs with relations and captions.

This script reads co-objects from an input JSON file, samples objects for each subject,
generates scene graphs with relations and captions using LLM, and saves results
with streaming output (real-time writing).

Usage:
    python run_generate_scene_graph.py \
        --input_path /path/to/coobjects.json \
        --output_path /path/to/output.json \
        --samples_per_subject 10 \
        --max_objects 3
"""

import os
import sys
import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import StreamGenerator
from src.generators.scene_graph_generator import (
    SceneGraphGenerator,
    GenerationTask,
    SceneGraphItem
)
from src.generators.scene_graph_generator.data_types import SyntheticDataset

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


class StreamingSceneGraphWriter:
    """Handles streaming output of scene graph results to file."""
    
    def __init__(self, output_path: str, dataset_name: str = "synthetic", existing_results: Optional[List[SceneGraphItem]] = None):
        """
        Initialize the streaming writer.
        
        Args:
            output_path: Path to the output JSON file
            dataset_name: Name of the dataset
            existing_results: Optional list of existing SceneGraphItem objects to preserve
        """
        self.output_path = Path(output_path)
        self.dataset_name = dataset_name
        self.lock = asyncio.Lock()
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing results if provided
        if existing_results:
            self.results: List[Dict[str, Any]] = [item.to_dict() for item in existing_results]
            logger.info(f"Loaded {len(self.results)} existing results from output file")
        else:
            self.results: List[Dict[str, Any]] = []
        
        # Write initial state
        self._write_to_file()
    
    def _write_to_file(self) -> None:
        """Write current results to file."""
        data = {
            "dataset": self.dataset_name,
            "total": len(self.results),
            "results": self.results
        }
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    async def add_result(self, result: SceneGraphItem) -> None:
        """
        Add a result and immediately write to file.
        
        Args:
            result: The SceneGraphItem to add
        """
        async with self.lock:
            result_dict = result.to_dict()
            result_id = result_dict["id"]
            
            # Check if ID already exists (shouldn't happen, but be safe)
            existing_index = next(
                (i for i, r in enumerate(self.results) if r["id"] == result_id),
                None
            )
            
            if existing_index is not None:
                # Update existing result instead of appending
                logger.warning(f"Result with ID {result_id} already exists, updating instead of appending")
                self.results[existing_index] = result_dict
            else:
                # Append new result
                self.results.append(result_dict)
            
            self._write_to_file()
    
    async def finalize(self) -> int:
        """
        Finalize the output file with sorted results.
        
        Returns:
            Total number of results
        """
        async with self.lock:
            # Sort results by ID to ensure sequential order
            self.results.sort(key=lambda x: x["id"])
            self._write_to_file()
            return len(self.results)


def load_existing_results(output_path: str) -> Optional[List[SceneGraphItem]]:
    """
    Load existing results from output file if it exists.
    
    Args:
        output_path: Path to the output JSON file
        
    Returns:
        List of existing SceneGraphItem objects, or None if file doesn't exist
    """
    path = Path(output_path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        dataset = SyntheticDataset.from_dict(data)
        logger.info(f"Loaded {len(dataset.results)} existing results from {output_path}")
        return dataset.results
    except Exception as e:
        logger.warning(f"Failed to load existing results from {output_path}: {e}")
        return None


def filter_missing_tasks(
    all_tasks: List[GenerationTask],
    existing_results: List[SceneGraphItem]
) -> Tuple[List[GenerationTask], Dict[int, SceneGraphItem], int]:
    """
    Filter out tasks that already have results and return missing tasks.
    
    Args:
        all_tasks: List of all GenerationTask objects
        existing_results: List of existing SceneGraphItem objects
        
    Returns:
        Tuple of (missing_tasks, existing_items_dict, next_id)
        - missing_tasks: Tasks that need to be generated
        - existing_items_dict: Dict mapping task_id to existing SceneGraphItem
        - next_id: Next available ID for new items
    """
    # Create a set of existing task IDs
    existing_ids = {item.id for item in existing_results}
    
    # Create dict mapping task_id to existing items
    existing_items_dict = {item.id: item for item in existing_results}
    
    # Find missing tasks
    missing_tasks = [task for task in all_tasks if task.task_id not in existing_ids]
    
    # Calculate next available ID
    next_id = max(existing_ids) + 1 if existing_ids else 0
    
    logger.info(
        f"Task analysis: {len(all_tasks)} total, {len(existing_ids)} existing, "
        f"{len(missing_tasks)} missing, next_id={next_id}"
    )
    
    return missing_tasks, existing_items_dict, next_id


async def generate_scene_graphs(
    generator: SceneGraphGenerator,
    tasks: List[GenerationTask],
    api_keys: list,
    output_path: str,
    model_name: str = "deepseek-chat",
    max_concurrent: int = 50,
    dataset_name: str = "synthetic",
    max_retries_per_task: int = 100,
    existing_results: Optional[List[SceneGraphItem]] = None
) -> int:
    """
    Generate scene graphs for all tasks using async streaming.
    
    Args:
        generator: The SceneGraphGenerator instance
        tasks: List of GenerationTask objects
        api_keys: List of API keys
        output_path: Path to the output JSON file
        model_name: Name of the LLM model to use
        max_concurrent: Maximum concurrent requests per key
        dataset_name: Name of the dataset
        max_retries_per_task: Maximum number of retries per failed task
        existing_results: Optional list of existing SceneGraphItem objects
        
    Returns:
        Number of successfully generated items
    """
    # Filter missing tasks if we have existing results
    if existing_results:
        missing_tasks, existing_items_dict, next_id = filter_missing_tasks(tasks, existing_results)
        
        if not missing_tasks:
            logger.info("All tasks already completed! No generation needed.")
            return len(existing_results)
        
        # Update task IDs for missing tasks to use next available IDs
        # But we need to maintain the mapping between original task_id and new item_id
        task_id_to_item_id = {}
        current_id = next_id
        
        for task in missing_tasks:
            task_id_to_item_id[task.task_id] = current_id
            current_id += 1
        
        tasks_to_generate = missing_tasks
        logger.info(f"Generating {len(tasks_to_generate)} missing tasks (IDs {next_id} to {current_id-1})")
    else:
        tasks_to_generate = tasks
        task_id_to_item_id = {task.task_id: task.task_id for task in tasks}
        existing_items_dict = {}
        logger.info(f"Generating all {len(tasks_to_generate)} tasks from scratch")
    
    # Initialize streaming writer with existing results
    writer = StreamingSceneGraphWriter(output_path, dataset_name, existing_results)
    
    # Initialize the stream generator
    stream_gen = StreamGenerator(
        model_name=model_name,
        api_keys=api_keys,
        max_concurrent_per_key=max_concurrent,
        max_retries=5,
        rational=False
    )
    
    # Prepare prompts with indices (use original task_id for tracking)
    prompts_with_index = [
        (task.task_id, generator.build_prompt(task.subject, task.sampled_objects))
        for task in tasks_to_generate
    ]
    
    # Create task lookup for building results
    task_lookup = {task.task_id: task for task in tasks_to_generate}
    
    # Create validators for each task
    validators = {
        task.task_id: generator.create_validator(task.subject, task.sampled_objects)
        for task in tasks_to_generate
    }
    
    system_prompt = generator.get_system_prompt()
    
    # Statistics
    success_count = 0
    failed_indices = set()
    completed_indices = set()  # Track completed task_ids to avoid duplicates
    
    logger.info(f"Starting generation for {len(tasks_to_generate)} tasks...")
    
    # Progress bar
    pbar = tqdm(total=len(tasks_to_generate), desc="Generating scene graphs", unit="task")
    
    # Process all prompts
    async for task_id, response in stream_gen.generate_stream_with_index(
        prompts_with_index=prompts_with_index,
        system_prompt=system_prompt,
        validate_func=None  # We validate manually per task
    ):
        # Skip if already completed (due to internal retries in StreamGenerator)
        if task_id in completed_indices:
            continue
        
        pbar.update(1)
        
        if response is None:
            task = task_lookup[task_id]
            logger.warning(f"Failed to generate for task {task_id}: {task.subject}")
            failed_indices.add(task_id)
            continue
        
        # Validate with task-specific validator
        validator = validators[task_id]
        validated_response = validator(response)
        
        if validated_response is None:
            task = task_lookup[task_id]
            logger.warning(f"Validation failed for task {task_id}: {task.subject}")
            failed_indices.add(task_id)
            continue
        
        # Build scene graph item with correct ID mapping
        task = task_lookup[task_id]
        item_id = task_id_to_item_id[task_id]
        item = generator.build_scene_graph_item(
            task=task,
            llm_response=validated_response,
            item_id=item_id
        )
        
        # Stream output - write immediately
        await writer.add_result(item)
        success_count += 1
        completed_indices.add(task_id)  # Mark as completed
        
        pbar.set_postfix({
            "success": success_count,
            "failed": len(failed_indices)
        })
    
    pbar.close()
    
    # Retry failed tasks with configurable max retries
    retry_round = 0
    while True:
        # Get tasks that still need to be completed
        retry_indices = failed_indices - completed_indices
        
        if not retry_indices:
            break
        
        retry_round += 1
        if retry_round > max_retries_per_task:
            logger.warning(f"Reached maximum retry rounds ({max_retries_per_task}), giving up on {len(retry_indices)} tasks")
            break
        
        logger.info(f"Retry round {retry_round}/{max_retries_per_task}: {len(retry_indices)} tasks remaining...")
        
        retry_prompts = [
            (task_id, generator.build_prompt(
                task_lookup[task_id].subject,
                task_lookup[task_id].sampled_objects
            ))
            for task_id in retry_indices
        ]
        
        retry_pbar = tqdm(
            total=len(retry_prompts), 
            desc=f"Retry round {retry_round}", 
            unit="task"
        )
        
        async for task_id, response in stream_gen.generate_stream_with_index(
            prompts_with_index=retry_prompts,
            system_prompt=system_prompt,
            validate_func=None
        ):
            # Skip if already completed
            if task_id in completed_indices:
                continue
            
            retry_pbar.update(1)
            
            if response is None:
                continue
            
            validator = validators[task_id]
            validated_response = validator(response)
            
            if validated_response is not None:
                task = task_lookup[task_id]
                item_id = task_id_to_item_id[task_id]
                item = generator.build_scene_graph_item(
                    task=task,
                    llm_response=validated_response,
                    item_id=item_id
                )
                
                await writer.add_result(item)
                success_count += 1
                completed_indices.add(task_id)  # Mark as completed
                failed_indices.discard(task_id)
                
                retry_pbar.set_postfix({
                    "success": success_count,
                    "remaining": len(retry_indices) - len(retry_indices & completed_indices)
                })
        
        retry_pbar.close()
        
        # Small delay between retry rounds
        if retry_indices - completed_indices:
            await asyncio.sleep(1)
    
    # Finalize output (sort by ID)
    total = await writer.finalize()
    
    final_failed = failed_indices - completed_indices
    logger.info(f"Successfully generated {success_count}/{len(tasks_to_generate)} new scene graphs")
    logger.info(f"Total items in output file: {total}")
    
    if final_failed:
        logger.warning(f"Permanently failed tasks ({len(final_failed)}): {list(final_failed)[:10]}...")
    
    return total


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate scene graphs with relations and captions using LLM"
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input co-objects JSON file"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output JSON file"
    )
    
    parser.add_argument(
        "--samples_per_subject",
        type=int,
        default=10,
        help="Number of samples to generate per subject (default: 10)"
    )
    
    parser.add_argument(
        "--min_objects",
        type=int,
        default=1,
        help="Minimum number of objects per scene (default: 1)"
    )
    
    parser.add_argument(
        "--max_objects",
        type=int,
        default=3,
        help="Maximum number of objects per scene (default: 3)"
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
        default=10,
        help="Maximum concurrent requests per API key (default: 10)"
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the key.yaml config file (default: config/key.yaml)"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="synthetic",
        help="Name of the dataset (default: synthetic)"
    )
    
    parser.add_argument(
        "--superclass",
        type=str,
        default=None,
        help="Superclass name to append after subject in captions (e.g., 'bird' for CUB-200)"
    )
    
    parser.add_argument(
        "--max_retries_per_task",
        type=int,
        default=100,
        help="Maximum number of retry rounds for failed tasks (default: 100)"
    )
    
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="mixed",
        choices=["mixed", "single_category"],
        help="Sampling strategy: 'mixed' (random from all categories, 1-3 objects) or "
             "'single_category' (from one category only) (default: mixed)"
    )

    parser.add_argument(
        "--sampling_category",
        type=str,
        default=None,
        choices=[
            "semantically_associated",
            "compatible_non_typical",
            "contextually_contrastive"
        ],
        help="Specific category to sample from in single_category mode "
             "(default: random non-empty category)"
    )
    
    parser.add_argument(
        "--objects_per_category",
        type=int,
        default=None,
        help="Number of objects to sample per category in single_category mode "
             "(defaults to a random value within min/max bounds)"
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
    output_path = Path(args.output_path)
    
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Samples per subject: {args.samples_per_subject}")
    logger.info(f"Objects per scene: {args.min_objects}-{args.max_objects}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Max retries per task: {args.max_retries_per_task}")
    logger.info(f"Sampling mode: {args.sampling_mode}")
    if args.sampling_category:
        logger.info(f"Sampling category: {args.sampling_category}")
    if args.objects_per_category:
        logger.info(f"Objects per category: {args.objects_per_category}")
    if args.superclass:
        logger.info(f"Superclass: {args.superclass}")
    
    # Load API keys
    api_keys = load_api_keys(str(config_path))
    
    # Initialize generator
    generator = SceneGraphGenerator(
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        max_retries=5,
        superclass=args.superclass,
        sampling_mode=args.sampling_mode,
        objects_per_category=args.objects_per_category,
        sampling_category=args.sampling_category
    )
    
    # Load co-objects
    coobjects = generator.load_coobjects(str(input_path))
    
    # Create generation tasks
    tasks = generator.create_generation_tasks(
        coobjects=coobjects,
        samples_per_subject=args.samples_per_subject
    )
    
    logger.info(f"Total tasks to generate: {len(tasks)}")
    
    # Check if output file exists and load existing results
    existing_results = None
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info(f"Output file exists: {output_path}")
        existing_results = load_existing_results(str(output_path))
        if existing_results:
            logger.info(f"Found {len(existing_results)} existing results, will generate missing ones only")
        else:
            logger.info("Could not load existing results, will generate from scratch")
    else:
        logger.info("Output file does not exist, will generate from scratch")
    
    # Run async generation
    total = asyncio.run(
        generate_scene_graphs(
            generator=generator,
            tasks=tasks,
            api_keys=api_keys,
            output_path=str(output_path),
            model_name=args.model_name,
            max_concurrent=args.max_concurrent,
            dataset_name=args.dataset_name,
            max_retries_per_task=args.max_retries_per_task,
            existing_results=existing_results
        )
    )
    
    logger.info(f"Generation completed! Total items: {total}")


if __name__ == "__main__":
    main()

