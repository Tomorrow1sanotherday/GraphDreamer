#!/bin/bash
# Script to run scene graph generation
#
# Usage:
#   ./run.sh [options]
#
# Examples:
#   ./run.sh --input_path /path/to/coobjects.json --output_path /path/to/output.json
#   ./run.sh --samples_per_subject 20 --max_objects 3
    # ALLOWED_CATEGORIES = (
    #     "semantically_associated",
    #     "compatible_non_typical",
    #     "contextually_contrastive"
    # )
# Unset all proxy environment variables to avoid httpx compatibility issues
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
INPUT_PATH="${PROJECT_ROOT}/data/cub2011/co_objects/coobjects_20.json"
OUTPUT_PATH="${PROJECT_ROOT}/data/cub2011/scene_graphs/scene_graphs_$(date +%Y%m%d_%H%M%S).json"
# OUTPUT_PATH="${PROJECT_ROOT}/data/cub2011/scene_graphs/scene_graphs_20260103_155125.json"
SAMPLES_PER_SUBJECT=30
MIN_OBJECTS=1
MAX_OBJECTS=1
MODEL_NAME="deepseek-v3"
MAX_CONCURRENT=10
CONFIG_PATH="${PROJECT_ROOT}/config/key.yaml"
DATASET_NAME="cub_2011_synthetic"
SUPERCLASS="bird"
MAX_RETRIES_PER_TASK=10
SAMPLING_MODE="single_category"
OBJECTS_PER_CATEGORY=""
SAMPLING_CATEGORY="semantically_associated"

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
        --samples_per_subject)
            SAMPLES_PER_SUBJECT="$2"
            shift 2
            ;;
        --min_objects)
            MIN_OBJECTS="$2"
            shift 2
            ;;
        --max_objects)
            MAX_OBJECTS="$2"
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
        --config_path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --superclass)
            SUPERCLASS="$2"
            shift 2
            ;;
        --max_retries_per_task)
            MAX_RETRIES_PER_TASK="$2"
            shift 2
            ;;
        --sampling_mode)
            SAMPLING_MODE="$2"
            shift 2
            ;;
        --sampling_category)
            SAMPLING_CATEGORY="$2"
            shift 2
            ;;
        --objects_per_category)
            OBJECTS_PER_CATEGORY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Scene Graph Generation"
echo "========================================"
echo "Input: $INPUT_PATH"
echo "Output: $OUTPUT_PATH"
echo "Samples per subject: $SAMPLES_PER_SUBJECT"
echo "Objects per scene: $MIN_OBJECTS-$MAX_OBJECTS"
echo "Model: $MODEL_NAME"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Max retries per task: $MAX_RETRIES_PER_TASK"
echo "Sampling mode: $SAMPLING_MODE"
if [ -n "$SAMPLING_CATEGORY" ]; then
    echo "Sampling category: $SAMPLING_CATEGORY"
fi
if [ -n "$OBJECTS_PER_CATEGORY" ]; then
    echo "Objects per category: $OBJECTS_PER_CATEGORY"
fi
echo "========================================"

# Build command arguments
CMD_ARGS=(
    --input_path "$INPUT_PATH"
    --output_path "$OUTPUT_PATH"
    --samples_per_subject "$SAMPLES_PER_SUBJECT"
    --min_objects "$MIN_OBJECTS"
    --max_objects "$MAX_OBJECTS"
    --model_name "$MODEL_NAME"
    --max_concurrent "$MAX_CONCURRENT"
    --config_path "$CONFIG_PATH"
    --dataset_name "$DATASET_NAME"
    --superclass "$SUPERCLASS"
    --max_retries_per_task "$MAX_RETRIES_PER_TASK"
    --sampling_mode "$SAMPLING_MODE"
)

# Add sampling_category if specified
if [ -n "$SAMPLING_CATEGORY" ]; then
    CMD_ARGS+=(--sampling_category "$SAMPLING_CATEGORY")
fi

# Add objects_per_category if specified
if [ -n "$OBJECTS_PER_CATEGORY" ]; then
    CMD_ARGS+=(--objects_per_category "$OBJECTS_PER_CATEGORY")
fi

# Run the Python script
python "${SCRIPT_DIR}/run_generate_scene_graph.py" "${CMD_ARGS[@]}"

