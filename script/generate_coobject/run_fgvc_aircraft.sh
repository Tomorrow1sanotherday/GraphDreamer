#!/bin/bash

# Script to run co-object generation for FGVC-Aircraft
# Usage: ./run_fgvc_aircraft.sh [n_objects] [model_name]

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_PATH="${PROJECT_ROOT}/data/fgvc_aircraft/labels.json"
OUTPUT_DIR="${PROJECT_ROOT}/data/fgvc_aircraft/co_objects"
N_OBJECTS="${1:-5}"
MODEL_NAME="${2:-deepseek-v4-flash}"

echo "====================================="
echo "Co-Object Generation - FGVC-Aircraft"
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
    --model_name "$MODEL_NAME" \
    --superclass "Aircraft"
