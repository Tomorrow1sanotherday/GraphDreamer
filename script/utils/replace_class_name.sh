#!/usr/bin/env bash
# 用法: bash replace_class_name.sh
# 所有参数在下方修改即可

# ========== 参数配置 ==========
INPUT_JSON="/mnt/sda/runhaofu/CopulaSyn_v2/data/cub2011/syn_images/co_object_50/syn_train_index_semantically_1_3.json"   # 必填: 输入 JSON 路径
OUTPUT_JSON=""                         # 可选: 输出路径。留空则与输入同目录，命名为 {输入名}_{NEW_NAME}_label{NEW_LABEL}.json
OLD_NAME="Parakeet_Auklet"                # 要替换的旧类名
NEW_NAME="rubbish"                     # 新类名
NEW_LABEL=200                          # 新标签 id
REPLACE_RATIO=1.0                      # 替换比例 0~1
SEED=42                                # 随机种子

# ========== 执行 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/replace_class_name.py"

ARGS=(
  "$INPUT_JSON"
  --old-name "$OLD_NAME"
  --new-name "$NEW_NAME"
  --new-label "$NEW_LABEL"
  --replace-ratio "$REPLACE_RATIO"
  --seed "$SEED"
)

if [[ -n "$OUTPUT_JSON" ]]; then
  ARGS+=(-o "$OUTPUT_JSON")
fi

exec python3 "$PYTHON_SCRIPT" "${ARGS[@]}"
