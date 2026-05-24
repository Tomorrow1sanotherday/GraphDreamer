#!/bin/bash
# Generate images from base_prompts.json for CUB-200 (photo).
# All parameters are defined below — edit and run.

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ---- IO ----
BASE_PROMPTS_PATH="${PROJECT_ROOT}/data/cub2011/base_prompts.json"
OUTPUT_DIR="${PROJECT_ROOT}/data/cub2011/syn_images/co_object_5/base_prompts_6000"
OUTPUT_JSON="${PROJECT_ROOT}/data/cub2011/syn_images/co_object_5/syn_train_index_base_6000.json"
DATASET_NAME="cub_2011_synthetic"

# ---- Model / generation ----
MODEL="stabilityai/stable-diffusion-xl-base-1.0"
DEVICE="cuda"
DTYPE="float16"
SEED=42
GUIDANCE_SCALE=""           # empty = model default; or set e.g. 7.5
NUM_INFERENCE_STEPS=50
IMAGE_WIDTH=1024
IMAGE_HEIGHT=1024
IMAGE_FORMAT="png"
FILENAME_PREFIX="syn_image"

# ---- Runtime ----
GPU_IDS="0,1,2,3,4,5,6,7"   # comma-separated GPU ids for multi-GPU
NO_RESUME=""                # set to "--no_resume" to regenerate everything
ENABLE_XFORMERS=""          # set to "--enable_xformers" to enable xformers
NO_SAFETENSORS=""           # set to "--no_safetensors" to disable safetensors
KEEP_INTERMEDIATE=""        # set to "--keep_intermediate" to retain the converted scene_graphs json

echo "========================================"
echo "Image Generation from Base Prompts (CUB-200)"
echo "========================================"
echo "Base prompts:       $BASE_PROMPTS_PATH"
echo "Output Directory:   $OUTPUT_DIR"
echo "Output JSON:        $OUTPUT_JSON"
echo "Dataset:            $DATASET_NAME"
echo "Model:              $MODEL"
echo "GPU IDs:            $GPU_IDS"
echo "Image Size:         ${IMAGE_WIDTH}x${IMAGE_HEIGHT}"
echo "Inference Steps:    $NUM_INFERENCE_STEPS"
echo "Image Format:       $IMAGE_FORMAT"
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
