#!/bin/bash
# Script to run attribute generation
#
# Usage:
#   ./run.sh [options]
#
# Examples:
#   ./run.sh --input_path /path/to/coobjects.json --output_dir /path/to/output_dir
#   ./run.sh --n_attributes 5 --superclass Bird

# Unset all proxy environment variables to avoid httpx compatibility issues
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
INPUT_PATH="${PROJECT_ROOT}/data/cub2011/co_objects/coobjects_20260103_001613.json"
OUTPUT_DIR="${PROJECT_ROOT}/data/cub2011/attributes"
N_ATTRIBUTES=5
MODEL_NAME="deepseek-v3"
MAX_CONCURRENT=5
CONFIG_PATH="${PROJECT_ROOT}/config/key.yaml"
SUPERCLASS="Bird"
MAX_RETRIES_PER_ITEM=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_path)
            INPUT_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --n_attributes)
            N_ATTRIBUTES="$2"
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
        --superclass)
            SUPERCLASS="$2"
            shift 2
            ;;
        --max_retries_per_item)
            MAX_RETRIES_PER_ITEM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Attribute Generation"
echo "========================================"
echo "Input: $INPUT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Attributes per subject: $N_ATTRIBUTES"
echo "Model: $MODEL_NAME"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Superclass: $SUPERCLASS"
echo "Max retries per item: $MAX_RETRIES_PER_ITEM"
echo "========================================"

# Build command arguments
CMD_ARGS=(
    --input_path "$INPUT_PATH"
    --output_dir "$OUTPUT_DIR"
    --n_attributes "$N_ATTRIBUTES"
    --model_name "$MODEL_NAME"
    --max_concurrent "$MAX_CONCURRENT"
    --config_path "$CONFIG_PATH"
    --superclass "$SUPERCLASS"
    --max_retries_per_item "$MAX_RETRIES_PER_ITEM"
)

# Run the Python script
python "${SCRIPT_DIR}/run_generate_attribute.py" "${CMD_ARGS[@]}"

