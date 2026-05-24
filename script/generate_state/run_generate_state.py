#!/usr/bin/env python3
"""
Script to generate subject states (pose, activity, condition) for each label class.

This script reads labels from an input JSON file, queries DeepSeek LLM
to generate diverse visual states that describe how the subject can appear in a photo.

Usage:
    python run_generate_state.py --input_path /path/to/labels.json --output_dir /path/to/output_dir --n_states 20
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
from src.generators.state_generator import SubjectStateGenerator
from src.generators.state_generator.generator import StateResult, LabelItem

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


async def generate_single_with_retry(
    stream_gen: StreamGenerator,
    generator: SubjectStateGenerator,
    label: LabelItem,
    index: int,
    superclass: str,
    max_retries: int = 100,
    progress_callback=None
) -> StateResult:
    """
    Generate states for a single label with retry on failure.

    Args:
        stream_gen: The StreamGenerator instance
        generator: The SubjectStateGenerator instance
        label: The label item to generate for
        index: The index of the label
        superclass: The broad category for all subjects
        max_retries: Maximum number of retries
        progress_callback: Optional callback function for progress updates

    Returns:
        StateResult for this label (guaranteed to succeed or raise error)
    """
    validator = generator.create_validator(label.display_name, superclass)
    prompt = generator.build_prompt(label.display_name, superclass)
    system_prompt = generator.get_system_prompt()

    for attempt in range(max_retries):
        try:
            if progress_callback:
                progress_callback(f"[{label.display_name}] Generating states (attempt {attempt + 1})...")

            async for _, response in stream_gen.generate_stream_with_index(
                prompts_with_index=[(index, prompt)],
                system_prompt=system_prompt,
                validate_func=None
            ):
                if response is None:
                    if progress_callback:
                        progress_callback(f"[{label.display_name}] attempt {attempt + 1} failed: no response")
                    continue

                validated_states = validator(response)

                if validated_states is None:
                    if progress_callback:
                        progress_callback(f"[{label.display_name}] attempt {attempt + 1} failed: validation error")
                    continue

                if progress_callback:
                    progress_callback(f"[{label.display_name}] ✓ completed ({len(validated_states)} states)")
                return StateResult(
                    label=label.label,
                    label_name=label.label_name,
                    subject=label.display_name,
                    states=validated_states
                )
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                if progress_callback:
                    progress_callback(f"[{label.display_name}] rate limited, waiting... (attempt {attempt + 1})")
            else:
                if progress_callback:
                    progress_callback(f"[{label.display_name}] error: {error_msg[:50]}")

        if attempt < max_retries - 1:
            delay = min(0.5 * (2 ** min(attempt, 5)), 10.0)
            await asyncio.sleep(delay)

    raise RuntimeError(
        f"Failed to generate states for {label.display_name} after {max_retries} attempts"
    )


async def update_partial_result(
    result: StateResult,
    output_path: Path,
    lock: asyncio.Lock,
    partial_results: dict
):
    """
    Write a single result to file and update partial results.

    Args:
        result: The StateResult to write
        output_path: Path to the output JSON file
        lock: Async lock for file writing
        partial_results: Dictionary to store partial results (keyed by label)
    """
    try:
        async with lock:
            partial_results[result.label] = result.to_dict()
            sorted_results = [partial_results[k] for k in sorted(partial_results.keys())]

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sorted_results, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            logger.info(
                f"✓ Written to file: {len(sorted_results)} total results, "
                f"label {result.label} ({result.subject}) with {len(result.states)} states"
            )
    except Exception as e:
        logger.error(f"Error writing result for label {result.label}: {e}", exc_info=True)
        raise


async def generate_states(
    generator: SubjectStateGenerator,
    labels: list,
    api_keys: list,
    output_path: Path,
    model_name: str = "deepseek-chat",
    max_concurrent: int = 50,
    superclass: str = "Bird",
    max_retries_per_label: int = 100
) -> list:
    """
    Generate states for all labels using async streaming.
    Results are written to file in real-time as each label completes.

    Args:
        generator: The SubjectStateGenerator instance
        labels: List of LabelItem objects
        api_keys: List of API keys
        output_path: Path to output file for streaming writes
        model_name: Name of the LLM model to use
        max_concurrent: Maximum concurrent requests per key
        superclass: The broad category for all subjects (e.g., "Bird")
        max_retries_per_label: Maximum retries per label

    Returns:
        List of StateResult objects
    """
    stream_gen = StreamGenerator(
        model_name=model_name,
        api_keys=api_keys,
        max_concurrent_per_key=max_concurrent,
        max_retries=5,
        rational=False
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([], f, indent=4, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    logger.info(f"Initialized output file: {output_path}")
    logger.info(f"Starting generation for {len(labels)} labels...")
    logger.info(f"Results will be written in real-time to: {output_path}")

    completed_labels = {"count": 0}
    lock = asyncio.Lock()
    file_lock = asyncio.Lock()
    partial_results = {}

    def progress_callback(msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}", flush=True)

    results = []
    pbar = tqdm(
        total=len(labels),
        desc="Labels",
        unit="label",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} labels [{elapsed}<{remaining}]",
        position=0,
        leave=True
    )

    tasks = []
    for i, label in enumerate(labels):
        async def task_wrapper(idx, lbl):
            def local_progress(msg: str):
                progress_callback(msg)

            try:
                result = await generate_single_with_retry(
                    stream_gen=stream_gen,
                    generator=generator,
                    label=lbl,
                    index=idx,
                    superclass=superclass,
                    max_retries=max_retries_per_label,
                    progress_callback=local_progress
                )

                await update_partial_result(
                    result, output_path, file_lock, partial_results
                )

                async with lock:
                    completed_labels["count"] += 1

                pbar.update(1)
                pbar.set_postfix({"labels": f"{completed_labels['count']}/{len(labels)}"})

                return result
            except Exception as e:
                async with lock:
                    completed_labels["count"] += 1
                pbar.update(1)
                raise e

        tasks.append(task_wrapper(i, label))

    try:
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                pbar.set_postfix({"labels": f"{completed_labels['count']}/{len(labels)}"})
            except Exception as e:
                pbar.close()
                logger.error(f"Error generating for a label: {e}")
                raise e
    finally:
        pbar.close()

    results.sort(key=lambda r: r.label)
    logger.info(f"Successfully generated all {len(results)} labels!")
    logger.info(f"Results saved to: {output_path}")

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate subject states (pose, activity, condition) for label classes using LLM"
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
        "--n_states",
        type=int,
        default=20,
        help="Number of states to generate per subject (default: 20)"
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
        "--max_retries_per_label",
        type=int,
        default=100,
        help="Maximum number of retries per label (default: 100)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.config_path is None:
        config_path = PROJECT_ROOT / "config" / "key.yaml"
    else:
        config_path = Path(args.config_path)

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"states_{timestamp}.json"
    output_path = output_dir / output_filename

    logger.info(f"Input file: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Number of states per subject: {args.n_states}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Superclass: {args.superclass}")

    api_keys = load_api_keys(str(config_path))

    generator = SubjectStateGenerator(
        n_states_per_subject=args.n_states,
        max_retries=10
    )

    labels = generator.load_labels(str(input_path))

    results = asyncio.run(
        generate_states(
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
