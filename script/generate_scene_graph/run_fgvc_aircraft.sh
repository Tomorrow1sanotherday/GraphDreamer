#!/bin/bash
# Run programmatic scene graph generation for FGVC-Aircraft.
#
# Usage:
#   ./run_fgvc_aircraft.sh [options]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

COOBJECTS_PATH="${PROJECT_ROOT}/data/fgvc_aircraft/co_objects/coobjects_5.json"
RELATIONS_PATH="${PROJECT_ROOT}/data/fgvc_aircraft/ralations.json"
OUTPUT_PATH="${PROJECT_ROOT}/data/fgvc_aircraft/scene_graphs/co_object_5/scene_graphs_6000_deepseekv4f.json"
STATES_PATH="${PROJECT_ROOT}/data/fgvc_aircraft/states/states_5.json"
STYLES_PATH="${PROJECT_ROOT}/data/fgvc_aircraft/styles.json"

SAMPLES_PER_SUBJECT=60
MIN_OBJECTS=1
MAX_OBJECTS=1
SAMPLING_CATEGORY="semantically_associated"
SUPERCLASS="aircraft"
SEED=42

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

[[ -n "$STATES_PATH" && -f "$STATES_PATH" ]] && CMD_ARGS+=(--states_path "$STATES_PATH")
[[ -n "$STYLES_PATH" && -f "$STYLES_PATH" ]] && CMD_ARGS+=(--styles_path "$STYLES_PATH")

echo "Running programmatic scene graph generation for FGVC-Aircraft..."
echo "  coobjects:  $COOBJECTS_PATH"
echo "  output:     $OUTPUT_PATH"
echo "  superclass: $SUPERCLASS"
echo "  samples/subject: $SAMPLES_PER_SUBJECT"

python "${SCRIPT_DIR}/run_generate_sg_programmtic.py" "${CMD_ARGS[@]}"
