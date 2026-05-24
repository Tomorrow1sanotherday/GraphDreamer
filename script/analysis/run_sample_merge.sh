#!/bin/bash
# 在 data/cub2011 下从三个 syn_train_index 按每类比例采样合并
# 合并顺序：先 semantically，再 compatible，再 contextually
# 用法示例：
#   ./run_sample_merge.sh
#   ./run_sample_merge.sh 0.7 0.2 0.1   # semantically 70%, compatible 20%, contextually 10%

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/../../data/cub2011" && pwd)"
cd "$DATA_DIR"

# R1=semantically, R2=compatible, R3=contextually
R1="${1:-0.9}"
R2="${2:-0.1}"
R3="${3:-0.0}"
OUT="syn_train_index_merged_${R1}_${R2}_${R3}.json"

python "$SCRIPT_DIR/sample_merge_syn_index.py" \
  --semantically /mnt/sda/runhaofu/DivSyn/data/cub2011/syn_images/co_object_5/syn_train_index_semantically_programmatic_1_1.json \
  --compatible /mnt/sda/runhaofu/DivSyn/data/cub2011/syn_images/co_object_5/syn_train_index_compatible_programmatic_1_1.json \
  --contextually /mnt/sda/runhaofu/DivSyn/data/cub2011/syn_images/co_object_5/syn_train_index_contextually_programmatic_1_1.json \
  --ratios "$R1" "$R2" "$R3" \
  -o "$OUT" \
  --seed 42

echo "Output: $DATA_DIR/$OUT"
