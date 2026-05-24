#!/bin/bash
# Script to run scene graph attributes addition
#
# Usage:
#   ./run_wa.sh [options]
#
# Examples:
#   ./run_wa.sh --scene_graphs_path /path/to/scene_graphs.json --attributes_path /path/to/attributes.json
#   ./run_wa.sh --scene_graphs_path /path/to/scene_graphs.json --attributes_path /path/to/attributes.json --output_path /path/to/output.json

# Unset all proxy environment variables to avoid httpx compatibility issues
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
SCENE_GRAPHS_PATH="${PROJECT_ROOT}/data/cub2011/scene_graphs/scene_graphs_20260104_154745.json"
ATTRIBUTES_PATH="${PROJECT_ROOT}/data/cub2011/attributes/coobjects_with_attributes_20260106_220827.json"
OUTPUT_PATH="${PROJECT_ROOT}/data/cub2011/scene_graphs/"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scene_graphs_path)
            SCENE_GRAPHS_PATH="$2"
            shift 2
            ;;
        --attributes_path)
            ATTRIBUTES_PATH="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Scene Graph Attributes Adder"
echo "========================================"
echo "Scene graphs input: $SCENE_GRAPHS_PATH"
echo "Attributes input: $ATTRIBUTES_PATH"
if [ -n "$OUTPUT_PATH" ]; then
    echo "Output: $OUTPUT_PATH"
else
    echo "Output: (auto-generated from input filename with 'wa' suffix)"
fi
echo "========================================"

# Build command arguments
CMD_ARGS=(
    --scene_graphs_path "$SCENE_GRAPHS_PATH"
    --attributes_path "$ATTRIBUTES_PATH"
)

# Add output_path if specified
if [ -n "$OUTPUT_PATH" ]; then
    CMD_ARGS+=(--output_path "$OUTPUT_PATH")
fi

# Run the Python script
python "${SCRIPT_DIR}/run_generate_scene_graph_wa.py" "${CMD_ARGS[@]}"

