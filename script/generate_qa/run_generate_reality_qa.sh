#!/bin/bash
# Generate reality QA from scene-graph JSON.
#
# Usage:
#   ./run_generate_reality_qa.sh [options]
#
# Examples:
#   ./run_generate_reality_qa.sh
#   ./run_generate_reality_qa.sh --input_path /path/to/scene_graphs.json
#   ./run_generate_reality_qa.sh --input_path /path/to/sg.json --output_dir /path/to/out

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default paths
INPUT_PATH="${PROJECT_ROOT}/data/cub2011/scene_graphs/co_object_5/150/scene_graphs_30000_deepseekv4f.json"
OUTPUT_DIR="${PROJECT_ROOT}/data/cub2011/scene_graphs/co_object_5/150/"

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
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Generate reality QA from scene-graph JSON."
            echo ""
            echo "Options:"
            echo "  --input_path   scene graphs JSON (default: scene_graphs_12000.json)"
            echo "  --output_dir   output directory (default: same as input)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CMD_ARGS=(
    --input_path "$INPUT_PATH"
    --output_dir "$OUTPUT_DIR"
)

echo "Generating reality QA..."
echo "  input:  $INPUT_PATH"
echo "  output: $OUTPUT_DIR"
echo ""

python "${SCRIPT_DIR}/run_generate_reality_qa.py" "${CMD_ARGS[@]}"
