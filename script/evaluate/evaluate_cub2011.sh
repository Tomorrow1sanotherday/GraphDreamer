#!/usr/bin/env bash
# 用法: bash evaluate_synthetic.sh
# 所有参数在下方修改即可

# ========== 参数配置 ==========
# 输入模式二选一（另一个留空）：
#   SYN_INDEX  — 指定 JSON 索引文件（DivSyn 生成器输出）
#   SYN_ROOT   — 直接指定图片目录（ImageFolder 格式，无需 JSON）
SYN_INDEX=""
SYN_ROOT="/mnt/sda/runhaofu/GraphDreamer/data/cub2011/syn_images/co_object_5/optimized_6000_deepseekv4f_images_v0"

N_PER_CLASS=3                               # 公平对比时每类真实样本数
SYN_SAMPLES=""                              # 合成样本上限，留空表示用全部
BATCH_SIZE=32                               # 特征提取 batch size
RESIZE=""                                   # 抽特征前将图片 resize 成 (RESIZE, RESIZE)，留空则用原图尺寸
NO_ZERO_SHOT=0                            # 1=跳过 CLIP zero-shot 评估
RESULTS_DIR="./results"                     # 结果保存目录
FEATURES_DIR="./features"                   # 特征缓存目录
DETAILED_RESULTS=1                          # 1=保存每样本预测到 CSV
SEED=42                                     # 随机种子
# CLIP backend (必填，三选一，无自动推断):
#   openai_clip — 官方 OpenAI clip 包，名字如 ViT-L/14
#   open_clip   — LAION/DataComp 等社区权重，必须同时填 CLIP_PRETRAINED
#   huggingface — transformers.CLIPModel，名字如 openai/clip-vit-large-patch14
CLIP_BACKEND="openai_clip"
# 模型名 (留空 = 沿用 config.yaml 的 model.vision_encoder)
# 格式取决于 CLIP_BACKEND:
#   openai_clip: RN50 / RN101 / ViT-B/16 / ViT-L/14 / ViT-L/14@336px
#   open_clip:   ViT-B-32 / ViT-L-14 / ViT-H-14 / ...(横线不是斜杠)
#   huggingface: openai/clip-vit-large-patch14 等 HF 标识
# 多个模型按顺序依次评估，每个模型一份结果
CLIP_MODELS=(
  "RN101"
  "ViT-B/16"
  "ViT-L/14"
  "ViT-L/14@336px"
)
# open_clip 预训练权重 tag (CLIP_BACKEND=open_clip 时必填)
# 常用: laion2b_s32b_b82k / laion400m_e32 / datacomp_xl_s13b_b90k
# 全部可用: python -c "import open_clip; print(open_clip.list_pretrained())"
CLIP_PRETRAINED=""
# 指定使用的 CUDA GPU；多卡用逗号分隔，如 "0,1"；留空则按默认顺序
GPU_ID="4"

# ========== 执行 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/evaluate_synthetic.py"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -n "$SYN_INDEX" && -n "$SYN_ROOT" ]]; then
  echo "Error: SYN_INDEX 和 SYN_ROOT 只能设置一个，请将另一个留空。"
  exit 1
fi
if [[ -z "$SYN_INDEX" && -z "$SYN_ROOT" ]]; then
  echo "Error: 请设置 SYN_INDEX（JSON 文件路径）或 SYN_ROOT（图片目录路径）。"
  exit 1
fi

ARGS=(
  --n_per_class "$N_PER_CLASS"
  --batch_size "$BATCH_SIZE"
  --results_dir "$RESULTS_DIR"
  --features_dir "$FEATURES_DIR"
  --seed "$SEED"
)

[[ -n "$SYN_INDEX" ]]          && ARGS+=(--syn_index "$SYN_INDEX")
[[ -n "$SYN_ROOT" ]]           && ARGS+=(--synthetic_root "$SYN_ROOT")
[[ -n "$SYN_SAMPLES" ]]        && ARGS+=(--syn_samples "$SYN_SAMPLES")
[[ -n "$RESIZE" ]]             && ARGS+=(--resize "$RESIZE")
if [[ -z "$CLIP_BACKEND" ]]; then
  echo "Error: CLIP_BACKEND 必填 (openai_clip / open_clip / huggingface)。"
  exit 1
fi
ARGS+=(--clip_backend "$CLIP_BACKEND")
[[ -n "$CLIP_PRETRAINED" ]]    && ARGS+=(--clip_pretrained "$CLIP_PRETRAINED")
[[ "$NO_ZERO_SHOT" = "1" ]]    && ARGS+=(--no_zero_shot)
[[ "$DETAILED_RESULTS" = "1" ]] && ARGS+=(--detailed_results)

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

cd "$PROJECT_ROOT" || exit 1
for CLIP_MODEL in "${CLIP_MODELS[@]}"; do
  echo "================================================================"
  echo "[$(date '+%F %T')] Running evaluation with CLIP_MODEL=${CLIP_MODEL}"
  echo "================================================================"
  RUN_ARGS=("${ARGS[@]}" --clip_model "$CLIP_MODEL")
  python3 "$PYTHON_SCRIPT" "${RUN_ARGS[@]}"
done
