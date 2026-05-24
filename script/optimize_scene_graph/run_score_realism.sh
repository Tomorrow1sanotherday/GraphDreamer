#!/bin/bash
# Score pre-generated reality-QA questions via LLM.
#
# Input:  reality_qa JSON (produced by run_generate_reality_qa.py)
# Output: realism_scores JSON (fed to greedy optimizer via --realism_path)
#
# Usage:
#   ./run_score_realism.sh [options]
#
# Examples:
#   ./run_score_realism.sh
#   ./run_score_realism.sh --input_path /path/to/reality_qa.json
#   ./run_score_realism.sh --model_name deepseek-v3 --batch_size 1000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default paths
INPUT_PATH="${PROJECT_ROOT}/data/cub2011/scene_graphs/co_object_5/reality_qa_12000_gemini.json"
OUTPUT_PATH="${PROJECT_ROOT}/data/cub2011/scene_graphs/co_object_5/realism_scores_12000_gemini.json"
CONFIG_PATH="${PROJECT_ROOT}/config/key.yaml"

# Scoring parameters
MODEL_NAME="deepseek-v4-flash"
MAX_CONCURRENT=20
FLUSH_EVERY=200
GROUP_QUESTIONS="false"
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
        --config_path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --max_concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --flush_every)
            FLUSH_EVERY="$2"
            shift 2
            ;;
        --group_questions)
            GROUP_QUESTIONS="true"
            shift
            ;;
        --no_group_questions)
            GROUP_QUESTIONS="false"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Score pre-generated reality-QA questions via LLM."
            echo ""
            echo "Options:"
            echo "  --input_path       reality_qa JSON (default: reality_qa_20260316_183822.json)"
            echo "  --output_path      output realism scores JSON"
            echo "  --config_path      key.yaml path (default: config/key.yaml)"
            echo "  --model_name       LLM model (default: deepseek-v3)"
            echo "  --max_concurrent   max concurrent requests per key (default: 300)"
            echo "  --flush_every      flush output every N completed records (default: 200)"
            echo "  --group_questions  score each record in one LLM request using a structured prompt"
            echo "  --no_group_questions  disable grouped scoring even if enabled in this script"
            echo "  --verbose          enable debug logging"
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
    --config_path "$CONFIG_PATH"
    --model_name "$MODEL_NAME"
    --max_concurrent "$MAX_CONCURRENT"
    --flush_every "$FLUSH_EVERY"
)

if [[ "$GROUP_QUESTIONS" == "true" ]]; then
    CMD_ARGS+=(--group_questions)
fi

if [[ -n "$VERBOSE" ]]; then
    CMD_ARGS+=($VERBOSE)
fi

echo "Running LLM realism scoring..."
echo "  input:           $INPUT_PATH"
echo "  output:          $OUTPUT_PATH"
echo "  model:           $MODEL_NAME"
echo "  max_concurrent:  $MAX_CONCURRENT"
echo "  flush_every:     $FLUSH_EVERY"
echo "  group_questions: $GROUP_QUESTIONS"
echo ""

python "${SCRIPT_DIR}/run_score_realism.py" "${CMD_ARGS[@]}"
