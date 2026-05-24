#!/usr/bin/env python3
"""
Script to generate reality QA from scene-graph JSON.

Produces feasibility-checking questions (R_SA, R_SO, R_SRO) for each
scene graph, intended for downstream zero-shot LLM scoring.

Usage:
    python script/generate_qa/run_generate_reality_qa.py \
        --input_path data/cub2011/scene_graphs/co_object_5/scene_graphs_12000.json \
        --output_dir data/cub2011/scene_graphs/co_object_5
"""

import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generators.qa_generator import RealityQAGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate reality QA from scene graphs for feasibility assessment"
    )

    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to the input scene-graph JSON file",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save the output JSON (default: same as input)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"reality_qa_{timestamp}.json"

    logger.info(f"Input file:  {input_path}")
    logger.info(f"Output file: {output_path}")

    generator = RealityQAGenerator()
    records = generator.process(str(input_path), str(output_path))

    logger.info(f"Done — {len(records)} scene graphs processed.")


if __name__ == "__main__":
    main()
