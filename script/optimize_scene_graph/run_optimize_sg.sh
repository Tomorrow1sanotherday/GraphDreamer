#!/bin/bash
# Run greedy submodular scene graph optimization.
#
# Selects K optimal scene graphs per class that maximize:
#   λ · D(G*) + (1−λ) · R̄(G*)
# where D is structural entropy diversity and R is realism score.
#
# Usage:
#   ./run_optimize_sg.sh [options]
#
# Examples:
#   ./run_optimize_sg.sh
#   ./run_optimize_sg.sh --budget_per_class 20 --lambda_weight 0.7
#   ./run_optimize_sg.sh --input_path /path/to/candidates.json --output_path /path/to/selected.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default paths
INPUT_PATH="${PROJECT_ROOT}/data/oxford_flowers102/scene_graphs/co_object_5/reality_qa_6120_deepseekv4f.json"
OUTPUT_PATH="${PROJECT_ROOT}/data/oxford_flowers102/scene_graphs/co_object_5/scene_graphs_optimized_6000_deepseekv4f_v0.json"

# Algorithm parameters
BUDGET_PER_CLASS=30
LAMBDA_WEIGHT=0.5
LOG_BASE="e"
VERBOSE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_path)
            INPUT_PATH="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --budget_per_class)
            BUDGET_PER_CLASS="$2"
            shift 2
            ;;
        --lambda_weight)
            LAMBDA_WEIGHT="$2"
            shift 2
            ;;
        --log_base)
            LOG_BASE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Greedy submodular scene graph optimization."
            echo "Selects K scene graphs per class maximizing λ·D(G*) + (1-λ)·R̄(G*)."
            echo ""
            echo "Options:"
            echo "  --input_path        candidate scene graphs JSON with realism_score"
            echo "  --output_path       output selected scene graphs JSON"
            echo "  --budget_per_class  K: number to select per class (default: 30)"
            echo "  --lambda_weight     λ ∈ [0,1]: 0=realism only, 1=diversity only (default: 0.5)"
            echo "  --log_base          entropy log base: e|2|10 (default: e)"
            echo "  --verbose           enable debug logging"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CMD_ARGS=(
    --input_path "$INPUT_PATH"
    --output_path "$OUTPUT_PATH"
    --budget_per_class "$BUDGET_PER_CLASS"
    --lambda_weight "$LAMBDA_WEIGHT"
    --log_base "$LOG_BASE"
)

if [[ -n "$VERBOSE" ]]; then
    CMD_ARGS+=($VERBOSE)
fi

echo "Running greedy scene graph optimization..."
echo "  input:         $INPUT_PATH"
echo "  output:        $OUTPUT_PATH"
echo "  budget/class:  $BUDGET_PER_CLASS"
echo "  lambda:        $LAMBDA_WEIGHT"
echo "  log_base:      $LOG_BASE"
echo ""

python "${SCRIPT_DIR}/run_optimize_sg.py" "${CMD_ARGS[@]}"
