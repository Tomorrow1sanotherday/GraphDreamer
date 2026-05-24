#!/bin/bash
# Generate images from scene graphs for FGVC-Aircraft
#
# Usage: ./run_fgvc_aircraft.sh [options]

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_PATH="${PROJECT_ROOT}/data/fgvc_aircraft/scene_graphs/co_object_5/scene_graphs_6000_deepseekv4f.json"
OUTPUT_DIR="${PROJECT_ROOT}/data/fgvc_aircraft/syn_images/co_object_5"
OUTPUT_JSON="${PROJECT_ROOT}/data/fgvc_aircraft/syn_images/co_object_5/syn_train_index_6000_deepseekv4f.json"
MODEL="stabilityai/stable-diffusion-xl-base-1.0"
DEVICE="cuda"
DTYPE="float16"
SEED=42
NUM_INFERENCE_STEPS=50
IMAGE_WIDTH=1024
IMAGE_HEIGHT=1024
IMAGE_FORMAT="png"
FILENAME_PREFIX="syn_image"
DATASET_NAME="fgvc_aircraft_synthetic"
GPU_IDS="4,5,6,7"

echo "========================================"
echo "Image Generation - FGVC-Aircraft"
echo "========================================"
echo "Input:              $INPUT_PATH"
echo "Output Directory:   $OUTPUT_DIR"
echo "Output JSON:        $OUTPUT_JSON"
echo "Model:              $MODEL"
echo "GPU IDs:            $GPU_IDS"
echo "========================================"

python "${SCRIPT_DIR}/run_generate_image.py" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --output_json "$OUTPUT_JSON" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --image_width "$IMAGE_WIDTH" \
    --image_height "$IMAGE_HEIGHT" \
    --image_format "$IMAGE_FORMAT" \
    --filename_prefix "$FILENAME_PREFIX" \
    --dataset_name "$DATASET_NAME" \
    ${GPU_IDS:+--gpu_ids "$GPU_IDS"}
