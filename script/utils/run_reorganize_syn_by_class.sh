#!/usr/bin/env bash
# 将 syn_images_semantically_1_1 的平铺图片重组为「每类一个文件夹」结构（类似 sdxl-baseprompt-30per）
# 用法: bash run_reorganize_syn_by_class.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYN_INDEX="/mnt/sda/runhaofu/DivSyn/data/cub2011/syn_images/co_object_20/syn_train_index_semantically_1_1.json"

# 默认复制图片（保留原文件夹）；若要移动并删除原图，加上 --move
python3 "$SCRIPT_DIR/reorganize_syn_images_by_class.py" \
  --syn_index "$SYN_INDEX" \
  "$@"

# 可选: 先试跑看会创建哪些目录
# python3 "$SCRIPT_DIR/reorganize_syn_images_by_class.py" --syn_index "$SYN_INDEX" --dry_run
