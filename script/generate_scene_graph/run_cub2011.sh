#!/bin/bash
# Run programmatic scene graph generation (no LLM).
#
# Usage:
#   ./run_generate_sg_programmtic.sh [options]
#
# Examples:
#   ./run_generate_sg_programmtic.sh
#   ./run_generate_sg_programmtic.sh --samples_per_subject 20 --max_objects 3
#   ./run_generate_sg_programmtic.sh --coobjects_path /path/to/coobjects_5.json --output_path /path/to/out.json
#   ./run_generate_sg_programmtic.sh --no-style --no-relation
#   ./run_generate_sg_programmtic.sh --no-coobject --no-state

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default paths
COOBJECTS_PATH="${PROJECT_ROOT}/data/cub2011/co_objects/coobjects_5_deepseekv4p.json"
RELATIONS_PATH="${PROJECT_ROOT}/data/cub2011/ralations.json"
OUTPUT_PATH="${PROJECT_ROOT}/data/cub2011/scene_graphs/co_object_5/150/scene_graphs_30000_deepseekv4f.json"
STATES_PATH="${PROJECT_ROOT}/data/cub2011/states/states_5.json"
STYLES_PATH="${PROJECT_ROOT}/data/cub2011/styles.json"

# Default generation params
SAMPLES_PER_SUBJECT=150
MIN_OBJECTS=1
MAX_OBJECTS=1
SAMPLING_CATEGORY="semantically_associated"
SUPERCLASS="bird"
SEED=42

# Component control flags
NO_STYLE=""
NO_RELATION=""
NO_COOBJECT=""
NO_STATE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --coobjects_path)
            COOBJECTS_PATH="$2"
            shift 2
            ;;
        --relations_path)
            RELATIONS_PATH="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --states_path)
            STATES_PATH="$2"
            shift 2
            ;;
        --styles_path)
            STYLES_PATH="$2"
            shift 2
            ;;
        --samples_per_subject)
            SAMPLES_PER_SUBJECT="$2"
            shift 2
            ;;
        --min_objects)
            MIN_OBJECTS="$2"
            shift 2
            ;;
        --max_objects)
            MAX_OBJECTS="$2"
            shift 2
            ;;
        --sampling_category)
            SAMPLING_CATEGORY="$2"
            shift 2
            ;;
        --superclass)
            SUPERCLASS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --no-style)
            NO_STYLE="--no-style"
            shift
            ;;
        --no-relation)
            NO_RELATION="--no-relation"
            shift
            ;;
        --no-coobject)
            NO_COOBJECT="--no-coobject"
            shift
            ;;
        --no-state)
            NO_STATE="--no-state"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  --coobjects_path   co-objects JSON (default: co_objects/coobjects_5.json)"
            echo "  --relations_path   relations JSON (default: ralations.json)"
            echo "  --output_path      output scene_graphs JSON"
            echo "  --states_path      optional states JSON for attributes.state"
            echo "  --styles_path      optional styles JSON for appending to caption"
            echo "  --samples_per_subject  (default: 10)"
            echo "  --min_objects      (default: 1)"
            echo "  --max_objects      (default: 2)"
            echo "  --sampling_category  semantically_associated|compatible_non_typical|contextually_contrastive"
            echo "  --superclass       (default: bird)"
            echo "  --seed             (default: 42)"
            echo "  --no-style         do not add style to caption"
            echo "  --no-relation      do not add relations to scene graph"
            echo "  --no-coobject      do not add co-objects to scene graph"
            echo "  --no-state         do not add state to scene graph attributes"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CMD_ARGS=(
    --coobjects_path "$COOBJECTS_PATH"
    --relations_path "$RELATIONS_PATH"
    --output_path "$OUTPUT_PATH"
    --samples_per_subject "$SAMPLES_PER_SUBJECT"
    --min_objects "$MIN_OBJECTS"
    --max_objects "$MAX_OBJECTS"
    --sampling_category "$SAMPLING_CATEGORY"
    --superclass "$SUPERCLASS"
    --seed "$SEED"
)

if [[ -n "$STATES_PATH" && -f "$STATES_PATH" ]]; then
    CMD_ARGS+=(--states_path "$STATES_PATH")
fi

if [[ -n "$STYLES_PATH" && -f "$STYLES_PATH" ]]; then
    CMD_ARGS+=(--styles_path "$STYLES_PATH")
fi

# Add component control flags if set
if [[ -n "$NO_STYLE" ]]; then
    CMD_ARGS+=($NO_STYLE)
fi

if [[ -n "$NO_RELATION" ]]; then
    CMD_ARGS+=($NO_RELATION)
fi

if [[ -n "$NO_COOBJECT" ]]; then
    CMD_ARGS+=($NO_COOBJECT)
fi

if [[ -n "$NO_STATE" ]]; then
    CMD_ARGS+=($NO_STATE)
fi

echo "Running programmatic scene graph generation..."
echo "  coobjects:  $COOBJECTS_PATH"
echo "  relations:  $RELATIONS_PATH"
echo "  output:     $OUTPUT_PATH"
echo "  states:     ${STATES_PATH:-none}"
echo "  styles:     ${STYLES_PATH:-none}"
echo "  samples/subject: $SAMPLES_PER_SUBJECT, objects: $MIN_OBJECTS-$MAX_OBJECTS"
echo "  flags:      style=${NO_STYLE:+disabled}${NO_STYLE:-enabled}, relation=${NO_RELATION:+disabled}${NO_RELATION:-enabled}, coobject=${NO_COOBJECT:+disabled}${NO_COOBJECT:-enabled}, state=${NO_STATE:+disabled}${NO_STATE:-enabled}"

python "${SCRIPT_DIR}/run_generate_sg_programmtic.py" "${CMD_ARGS[@]}"
