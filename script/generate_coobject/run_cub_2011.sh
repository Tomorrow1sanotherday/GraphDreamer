#!/bin/bash

# Script to run the co-object generation
# Usage: ./run.sh [n_objects]

# Unset all proxy environment variables to avoid httpx compatibility issues
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
INPUT_PATH="${PROJECT_ROOT}/data/cub2011/labels.json"
OUTPUT_DIR="${PROJECT_ROOT}/data/cub2011/co_objects"
N_OBJECTS="${1:-5}"
MODEL_NAME="${2:-deepseek-v4-flash}"
# MODEL_NAME="${2:-gemini-3.1-pro-preview}"

echo "====================================="
echo "Co-Object Generation Script"
echo "====================================="
echo "Input: ${INPUT_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Objects per subject: ${N_OBJECTS}"
echo "Model: ${MODEL_NAME}"
echo "====================================="

cd "$PROJECT_ROOT"

python script/generate_coobject/run_generate_coobject.py \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_objects "$N_OBJECTS" \
    --model_name "$MODEL_NAME"

