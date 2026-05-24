#!/bin/bash
# Build an evaluation-ready synthetic subset from optimized scene graph selections.
#
# Input:
#   1. optimized scene-graph JSON with selection_summary[].selected_ids
#   2. source syn_train_index JSON with original prompts and image paths
#
# Output:
#   1. filtered syn_train_index JSON
#   2. new image directory, preserving original selected image file names
#
# Usage:
#   ./run_build_optimized_eval_subset.sh [options]
#
# Examples:
#   ./run_build_optimized_eval_subset.sh --dry_run
#   ./run_build_optimized_eval_subset.sh --overwrite
#   ./run_build_optimized_eval_subset.sh --copy_mode symlink

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default paths
# OPTIMIZED_JSON="${PROJECT_ROOT}/data/stanford_car/scene_graphs/co_object_5/optimized_scene_graphs_5880.json"
# SOURCE_INDEX_JSON="${PROJECT_ROOT}/data/stanford_car/syn_images/co_object_5/syn_train_index_11760_with_quality.json"
OPTIMIZED_JSON="${PROJECT_ROOT}/data/oxford_flowers102/scene_graphs/co_object_5/scene_graphs_optimized_3060_deepseekv4f_v0.json"
SOURCE_INDEX_JSON="${PROJECT_ROOT}/data/oxford_flowers102/syn_images/co_object_5/syn_train_index_6120_deepseekv4f.json"
OUTPUT_JSON="/mnt/sda/runhaofu/GraphDreamer/data/oxford_flowers102/syn_images/co_object_5/scene_graphs_optimized_3060_deepseekv4f_v0.json"
OUTPUT_IMAGE_DIR="/mnt/sda/runhaofu/GraphDreamer/data/oxford_flowers102/syn_images/co_object_5/scene_graphs_optimized_3060_deepseekv4f_v0_images"

# Materialization parameters
COPY_MODE="copy"
OVERWRITE="false"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --optimized_json)
            OPTIMIZED_JSON="$2"
            shift 2
            ;;
        --source_index_json)
            SOURCE_INDEX_JSON="$2"
            shift 2
            ;;
        --output_json)
            OUTPUT_JSON="$2"
            shift 2
            ;;
        --output_image_dir)
            OUTPUT_IMAGE_DIR="$2"
            shift 2
            ;;
        --copy_mode)
            COPY_MODE="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE="true"
            shift
            ;;
        --dry_run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Build a syn_train_index subset and image folder from optimized scene-graph selections."
            echo ""
            echo "Options:"
            echo "  --optimized_json     optimized scene graph JSON"
            echo "  --source_index_json  source syn_train_index JSON"
            echo "  --output_json        output syn_train_index JSON path"
            echo "  --output_image_dir   output image directory"
            echo "  --copy_mode          copy|symlink (default: copy)"
            echo "  --overwrite          replace existing output JSON and image directory"
            echo "  --dry_run            validate inputs and preview file operations without writing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CMD_ARGS=(
    --optimized_json "$OPTIMIZED_JSON"
    --source_index_json "$SOURCE_INDEX_JSON"
    --copy_mode "$COPY_MODE"
)

if [[ -n "$OUTPUT_JSON" ]]; then
    CMD_ARGS+=(--output_json "$OUTPUT_JSON")
fi

if [[ -n "$OUTPUT_IMAGE_DIR" ]]; then
    CMD_ARGS+=(--output_image_dir "$OUTPUT_IMAGE_DIR")
fi

if [[ "$OVERWRITE" == "true" ]]; then
    CMD_ARGS+=(--overwrite)
fi

if [[ "$DRY_RUN" == "true" ]]; then
    CMD_ARGS+=(--dry_run)
fi

echo "Building optimized evaluation subset..."
echo "  optimized_json:   $OPTIMIZED_JSON"
echo "  source_index:     $SOURCE_INDEX_JSON"
if [[ -n "$OUTPUT_JSON" ]]; then
    echo "  output_json:      $OUTPUT_JSON"
fi
if [[ -n "$OUTPUT_IMAGE_DIR" ]]; then
    echo "  output_image_dir: $OUTPUT_IMAGE_DIR"
fi
echo "  copy_mode:        $COPY_MODE"
echo "  overwrite:        $OVERWRITE"
echo "  dry_run:          $DRY_RUN"
echo ""

python "$SCRIPT_DIR/build_optimized_eval_subset.py" "${CMD_ARGS[@]}"