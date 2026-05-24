#!/bin/bash
# Script to generate images from scene graphs for the CUB-2011 painting variant.
# Mirrors run_cub2011.sh but reads scene graphs from data/cub2011_painting/...
# and writes synthetic images under data/cub2011_painting/syn_images/...
#
# Usage:
#   ./run_cub2011_painting.sh [options]
#
# Examples:
#   ./run_cub2011_painting.sh --input_path /path/to/scene_graphs.json --output_dir /path/to/images
#   ./run_cub2011_painting.sh --model stabilityai/stable-diffusion-xl-base-1.0 --image_format png

# Unset all proxy environment variables to avoid compatibility issues
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values (cub2011_painting variant)
DATA_DIR="${PROJECT_ROOT}/data/cub2011_painting"
INPUT_PATH="${DATA_DIR}/scene_graphs/co_object_5/150/scene_graphs_30000_deepseekv4f.json"
OUTPUT_DIR="${DATA_DIR}/syn_images/co_object_5/scene_graphs_30000_deepseekv4f"
OUTPUT_JSON="${DATA_DIR}/syn_images/co_object_5/syn_train_index_30000_deepseekv4f.json"
MODEL="stabilityai/stable-diffusion-xl-base-1.0"
DEVICE="cuda"
DTYPE="float16"
SEED=42
GUIDANCE_SCALE=""  # Leave empty to use model default, or set a value like 7.5
NUM_INFERENCE_STEPS=50
IMAGE_WIDTH=1024
IMAGE_HEIGHT=1024
IMAGE_FORMAT="png"
FILENAME_PREFIX="syn_image"
DATASET_NAME="cub_2011_painting_synthetic"
NO_RESUME=""
ENABLE_XFORMERS=""
NO_SAFETENSORS=""
GPU_IDS="4,5,6,7"

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
        --output_json)
            OUTPUT_JSON="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --guidance_scale)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --num_inference_steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --image_width)
            IMAGE_WIDTH="$2"
            shift 2
            ;;
        --image_height)
            IMAGE_HEIGHT="$2"
            shift 2
            ;;
        --image_format)
            IMAGE_FORMAT="$2"
            shift 2
            ;;
        --filename_prefix)
            FILENAME_PREFIX="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --no_resume)
            NO_RESUME="--no_resume"
            shift
            ;;
        --enable_xformers)
            ENABLE_XFORMERS="--enable_xformers"
            shift
            ;;
        --no_safetensors)
            NO_SAFETENSORS="--no_safetensors"
            shift
            ;;
        --gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --input_path PATH         Input scene graphs JSON file"
            echo "  --output_dir PATH         Directory to save generated images"
            echo "  --output_json PATH        Output JSON file with image paths"
            echo "  --model NAME              Model name (default: stabilityai/stable-diffusion-xl-base-1.0)"
            echo "  --device DEVICE           Device to use (default: cuda)"
            echo "  --dtype DTYPE             Torch dtype (default: float16)"
            echo "  --seed SEED               Random seed (default: 42)"
            echo "  --guidance_scale SCALE    Guidance scale (default: 7.5)"
            echo "  --num_inference_steps N   Number of inference steps (default: 50)"
            echo "  --image_width WIDTH       Image width (default: 1024)"
            echo "  --image_height HEIGHT     Image height (default: 1024)"
            echo "  --image_format FORMAT     Image format: jpg or png (default: png)"
            echo "  --filename_prefix PREFIX  Filename prefix (default: syn_image)"
            echo "  --dataset_name NAME       Dataset name (default: cub_2011_painting_synthetic)"
            echo "  --no_resume               Disable resume mode (streaming output always enabled)"
            echo "  --enable_xformers         Enable xformers memory optimization"
            echo "  --no_safetensors          Disable safetensors loading"
            echo "  --gpu_ids IDS             GPU ID(s) to use. Can be:"
            echo "                            - Single GPU: '0' or '2' (e.g., --gpu_ids 2)"
            echo "                            - Multiple GPUs: '0,1,2' for parallel generation"
            echo "                            If not specified, automatically extracts GPU ID from --device (e.g., cuda:2 -> GPU 2)"
            echo "  -h, --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Image Generation from Scene Graphs (cub2011_painting)"
echo "========================================"
echo "Input:              $INPUT_PATH"
echo "Output Directory:   $OUTPUT_DIR"
echo "Output JSON:        $OUTPUT_JSON"
echo "Model:              $MODEL"
echo "Device:             $DEVICE"
echo "Dtype:              $DTYPE"
echo "Seed:               $SEED"
if [ -n "$GUIDANCE_SCALE" ]; then
    echo "Guidance Scale:     $GUIDANCE_SCALE"
else
    echo "Guidance Scale:     (using model default)"
fi
echo "Inference Steps:    $NUM_INFERENCE_STEPS"
echo "Image Size:         ${IMAGE_WIDTH}x${IMAGE_HEIGHT}"
echo "Image Format:       $IMAGE_FORMAT"
if [ -n "$GPU_IDS" ]; then
    echo "GPU IDs:            $GPU_IDS"
fi
echo "========================================"

# Run the Python script
python "${SCRIPT_DIR}/run_generate_image.py" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --output_json "$OUTPUT_JSON" \
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
    --dataset_name "$DATASET_NAME" \
    $NO_RESUME \
    $ENABLE_XFORMERS \
    $NO_SAFETENSORS \
    ${GPU_IDS:+--gpu_ids "$GPU_IDS"}
