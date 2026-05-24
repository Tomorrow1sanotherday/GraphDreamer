#!/bin/bash

# Script to run the subject state generation for Stanford Dogs
# Usage: ./run_stanford_dogs.sh [n_states] [model_name]

# Unset all proxy environment variables to avoid httpx compatibility issues
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
INPUT_PATH="${PROJECT_ROOT}/data/stanford_dogs/labels.json"
OUTPUT_DIR="${PROJECT_ROOT}/data/stanford_dogs/states"
N_STATES="${1:-5}"
MODEL_NAME="${2:-deepseek-v4-flash}"

echo "====================================="
echo "Subject State Generation - Stanford Dogs"
echo "====================================="
echo "Input: ${INPUT_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "States per subject: ${N_STATES}"
echo "Model: ${MODEL_NAME}"
echo "====================================="

cd "$PROJECT_ROOT"

python script/generate_state/run_generate_state.py \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_states "$N_STATES" \
    --model_name "$MODEL_NAME" \
    --superclass "Dog"
