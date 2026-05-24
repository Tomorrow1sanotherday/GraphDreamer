"""
LLM-based Structural Realism Scoring

Sends pre-generated reality-QA questions to LLM and collects scores.
Input:  a reality_qa JSON (produced by run_generate_reality_qa.py)
Output: a realism_scores JSON (fed to the greedy optimizer via --realism_path)

Each record's questions are scored independently by the LLM in [0, 1]:
    R(g) = mean of all dimension scores (R_SA, R_SO, R_SRO, R_SCENE, ...)

Usage:
    python run_score_realism.py \\
        --input_path  data/cub2011/scene_graphs/co_object_5/reality_qa_xxx.json \\
        --output_path data/cub2011/scene_graphs/co_object_5/realism_scores.json
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.api import StreamGenerator  # noqa: E402
from src.metrics.realism import RealismScorer  # noqa: E402

logger = logging.getLogger(__name__)


def load_api_keys(config_path: str) -> list:
    """Load API keys from key.yaml."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    keys = config.get("keys", [])
    if not keys:
        raise ValueError(f"No API keys found in {config_path}")
    logger.info("Loaded %d API keys from %s", len(keys), config_path)
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score pre-generated reality-QA questions via LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to reality_qa JSON (with pre-built questions per record)",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to output realism scores JSON",
    )
    parser.add_argument(
        "--config_path", type=str, default=None,
        help="Path to key.yaml (default: config/key.yaml)",
    )
    parser.add_argument(
        "--model_name", type=str, default="deepseek-v3",
        help="LLM model name (default: deepseek-v3)",
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=300,
        help="Max concurrent requests per API key (default: 300)",
    )
    parser.add_argument(
        "--flush_every", type=int, default=200,
        help="Flush output file every N completed records for real-time viewing (default: 200)",
    )
    parser.add_argument(
        "--group_questions", action="store_true",
        help=(
            "Score all questions of one record in a single LLM request using a structured prompt. "
            "This reduces API calls and improves repeated-prefix reuse, but can change the score distribution."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug-level logging",
    )
    return parser.parse_args()


async def run_scoring(args: argparse.Namespace) -> None:
    config_path = args.config_path or str(PROJECT_ROOT / "config" / "key.yaml")
    api_keys = load_api_keys(config_path)

    stream_gen = StreamGenerator(
        model_name=args.model_name,
        api_keys=api_keys,
        max_concurrent_per_key=args.max_concurrent,
        max_retries=5,
        rational=False,
    )

    scorer = RealismScorer(stream_gen)

    t0 = time.time()
    results = await scorer.score_qa_file(
        qa_path=args.input_path,
        output_path=args.output_path,
        flush_every=args.flush_every,
        group_questions=args.group_questions,
        show_progress=True,
    )
    elapsed = time.time() - t0

    avg_score = sum(r.realism_score for r in results) / len(results) if results else 0
    n_questions = sum(len(r.dimension_scores) for r in results)
    print(f"\n{'='*60}")
    print(f"  Realism Scoring — done in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  Records:          {len(results)}")
    print(f"  Questions scored: {n_questions}")
    print(f"  Model:            {args.model_name}")
    print(f"  Group questions:  {args.group_questions}")
    print(f"  Avg realism:      {avg_score:.4f}")
    print(f"  Output:           {args.output_path}")
    print(f"{'='*60}\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    asyncio.run(run_scoring(args))


if __name__ == "__main__":
    main()
