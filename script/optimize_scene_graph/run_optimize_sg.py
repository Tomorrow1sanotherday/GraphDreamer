"""
Greedy Scene Graph Optimization

Selects an optimal subset of scene graphs per class using greedy submodular
maximization that balances structural entropy diversity and realism.

    max_{G*}  λ · D(G*) + (1−λ) · R̄(G*)     s.t. |G*| = K

Usage:
    python run_optimize_sg.py \\
        --input_path  data/cub2011/scene_graphs/co_object_5/scene_graphs_12000.json \\
        --output_path data/cub2011/scene_graphs/co_object_5/scene_graphs_optimized.json \\
        --budget_per_class 30 \\
        --lambda_weight 0.5
"""

import argparse
import logging
import os
import sys
import time
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Register src and src.metrics as namespace packages so that the heavyweight
# src/__init__.py (which pulls in openai, diffusers, etc.) is not executed.
_src_dir = PROJECT_ROOT / "src"
for _pkg, _dir in [("src", _src_dir), ("src.metrics", _src_dir / "metrics")]:
    if _pkg not in sys.modules:
        _m = types.ModuleType(_pkg)
        _m.__path__ = [str(_dir)]
        _m.__package__ = _pkg
        sys.modules[_pkg] = _m

from src.metrics.selection import GreedySceneGraphSelector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy submodular scene graph subset selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to candidate scene graphs JSON",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to output selected scene graphs JSON",
    )
    parser.add_argument(
        "--budget_per_class", type=int, default=30,
        help="Number of scene graphs to select per class (default: 30)",
    )
    parser.add_argument(
        "--lambda_weight", type=float, default=0.5,
        help="Trade-off weight in [0,1]: 0=pure realism, 1=pure diversity (default: 0.5)",
    )
    parser.add_argument(
        "--log_base", type=str, default="e", choices=["e", "2", "10"],
        help="Entropy logarithm base (default: e)",
    )
    parser.add_argument(
        "--flush_every", type=int, default=10,
        help="Write output file every N classes for real-time viewing (default: 10)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug-level logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    selector = GreedySceneGraphSelector(
        budget_per_class=args.budget_per_class,
        lambda_weight=args.lambda_weight,
        log_base=args.log_base,
    )

    t0 = time.time()
    results = selector.select_from_file(
        input_path=args.input_path,
        output_path=args.output_path,
        flush_every=args.flush_every,
    )
    elapsed = time.time() - t0

    n_cls = len(results)
    total_selected = sum(r.num_selected for r in results)
    avg_d = sum(r.diversity for r in results) / n_cls
    avg_r = sum(r.avg_realism for r in results) / n_cls
    avg_o = sum(r.objective for r in results) / n_cls

    print(f"\n{'='*60}")
    print(f"  Greedy Scene Graph Optimization — done in {elapsed:.2f}s")
    print(f"{'='*60}")
    print(f"  Classes:          {n_cls}")
    print(f"  Budget/class:     {args.budget_per_class}")
    print(f"  Total selected:   {total_selected}")
    print(f"  Lambda:           {args.lambda_weight}")
    print(f"  Log base:         {args.log_base}")
    print(f"  Avg diversity:    {avg_d:.4f}")
    print(f"  Avg realism:      {avg_r:.4f}")
    print(f"  Avg objective:    {avg_o:.4f}")
    print(f"  Output:           {args.output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
