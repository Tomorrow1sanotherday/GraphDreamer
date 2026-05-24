#!/bin/bash
# Generate images from base_prompts.json for Stanford Cars.

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ---- IO ----
BASE_PROMPTS_PATH="${PROJECT_ROOT}/data/stanford_car/base_prompts_v3.json"
OUTPUT_DIR="${PROJECT_ROOT}/data/stanford_car/syn_images/base_prompts_5880_v3"
OUTPUT_JSON="${PROJECT_ROOT}/data/stanford_car/syn_images/syn_train_index_base_5880_v3.json"
DATASET_NAME="stanford_car_synthetic"

# ---- Model / generation ----
MODEL="stabilityai/stable-diffusion-xl-base-1.0"
DEVICE="cuda"
DTYPE="float16"
SEED=42
GUIDANCE_SCALE=""
NUM_INFERENCE_STEPS=50
IMAGE_WIDTH=1024
IMAGE_HEIGHT=1024
IMAGE_FORMAT="png"
FILENAME_PREFIX="syn_image"

# ---- Runtime ----
GPU_IDS="6,7"
NO_RESUME=""
ENABLE_XFORMERS=""
NO_SAFETENSORS=""
KEEP_INTERMEDIATE=""

echo "========================================"
echo "Image Generation from Base Prompts (Stanford Cars)"
echo "========================================"
echo "Base prompts:       $BASE_PROMPTS_PATH"
echo "Output Directory:   $OUTPUT_DIR"
echo "Output JSON:        $OUTPUT_JSON"
echo "Dataset:            $DATASET_NAME"
echo "GPU IDs:            $GPU_IDS"
echo "========================================"

python "${SCRIPT_DIR}/run_generate_base_image.py" \
    --base_prompts_path "$BASE_PROMPTS_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --output_json "$OUTPUT_JSON" \
    --dataset_name "$DATASET_NAME" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    ${GUIDANCE_SCALE:+--guidance_scale "$GUIDANCE_SCALE"} \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --image_width "$IMAGE_WIDTH" \
    --image_height "$IMAGE_HEIGHT" \
    --image_format "$IMAGE_FORMAT" \
    --filename_prefix "$FILENAME_PREFIX" \
    --gpu_ids "$GPU_IDS" \
    $NO_RESUME \
    $ENABLE_XFORMERS \
    $NO_SAFETENSORS \
    $KEEP_INTERMEDIATE
