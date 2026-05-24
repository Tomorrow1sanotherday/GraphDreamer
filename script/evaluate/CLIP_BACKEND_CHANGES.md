# 评测支持多 CLIP backbone — 改动记录

让 `script/evaluate/` 支持 **三种** CLIP backbone：
1. **OpenAI clip 包**（`import clip`）— 9 个原版权重: `RN50/101/50x4/16/64`, `ViT-B/32/16`, `ViT-L/14`, `ViT-L/14@336px`
2. **HuggingFace transformers**（`CLIPModel`）— HF Hub 上的 CLIP 权重，例如 `openai/clip-vit-large-patch14`
3. **open_clip**（`import open_clip`）— LAION / DataComp / 其它社区训练的大规模 CLIP 权重

## 背景

- 旧实现只用 HuggingFace `transformers.CLIPModel`，受限于 HF Hub 上有的权重（ViT 变体 OK，ResNet 变体 RN50/RN101 没有）。
- AttrSyn 用的是 OpenAI 官方 `clip` 包（`import clip; clip.load(...)`），9 个名字都原生支持。
- 想用 LAION-2B / DataComp 训的更强 CLIP（如 ViT-L-14/laion2b_s32b_b82k、ViT-H-14、ViT-bigG-14）需要 open_clip 包。
- `copulasyn` 环境里 `clip` 和 `open_clip`（v3.1.0+）都已装好。

## 改了什么

### `src/evaluator/feature_extractor.py`

`CLIPFeatureExtractor.__init__(model_name, device=None, pretrained=None, backend=None)`：

- **`backend` 显式传入** → 直接走指定 backend，跳过自动推断。值必须是 `openai_clip` / `open_clip` / `huggingface`；`open_clip` 时 `pretrained` 必填，否则 raise `ValueError`。`script/evaluate/` 全部走这条路径。
- **`backend=None`（向后兼容）** → fallback 到自动推断（`pretrained` 非空 → `open_clip`，名字命中 `OPENAI_CLIP_NAMES` → `openai_clip`，否则 `huggingface`）。`experiments/clip_selection.py`、`experiments/grid_zsfilter.py`、`src/metrics/quality.py` 等历史调用方走这条。

三个 backend 各自怎么干活：

| backend | 加载 API | 名字格式 | pretrained |
| --- | --- | --- | --- |
| `openai_clip` | `clip.load(name, device)` | `ViT-L/14`（斜杠） | 不用 |
| `open_clip` | `open_clip.create_model_and_transforms(name, pretrained, device)` | `ViT-L-14`（横线） | **必填** |
| `huggingface` | `CLIPModel.from_pretrained(name)` + `CLIPProcessor` | `openai/clip-vit-large-patch14` | 不用 |

`encode_images_normalized()` helper 返回 L2 归一化的 `torch.Tensor`，给 zero-shot 评测使用，屏蔽 backend 差异。

### `src/evaluator/clip_evaluator.py`

zero-shot 评测不再直接戳 `processor` / `get_image_features`，改调 `extractor.encode_images_normalized()`。三个 backend 都兼容。

### `script/evaluate/evaluate_synthetic.py`

- **`--clip_backend`（必填）**：`openai_clip` / `open_clip` / `huggingface` 三选一。**无自动推断**——不填 argparse 直接报错。写入 `config.model.vision_encoder_backend` 后透传给 extractor。
- `--clip_model`（可选）：覆盖 `config.model.vision_encoder`。格式取决于 backend（见上表）
- `--clip_pretrained`（`--clip_backend=open_clip` 时必填）：open_clip 权重 tag，写入 `config.model.vision_encoder_pretrained`
- test-feature 缓存路径加上 backbone slug **+ pretrained slug**，**避免不同模型/权重复用对方的 cache**

### `script/evaluate/evaluate_*.sh` (5 个脚本)

`evaluate_synthetic.sh` / `evaluate_cub2011_painting.sh` / `evaluate_fgvc_aircraft.sh` / `evaluate_oxford_flowers102.sh` / `evaluate_stanford_car.sh` 顶部都加了：

```bash
# CLIP backend (必填，三选一，无自动推断)
CLIP_BACKEND="openai_clip"
# 模型名 (留空 = 沿用 config.yaml 的 model.vision_encoder)
# 格式取决于 CLIP_BACKEND:
#   openai_clip: RN50 / RN101 / ViT-B/16 / ViT-L/14 / ViT-L/14@336px
#   open_clip:   ViT-B-32 / ViT-L-14 / ViT-H-14 / ...(横线不是斜杠)
#   huggingface: openai/clip-vit-large-patch14 等 HF 标识
CLIP_MODEL="ViT-L/14"
# open_clip 预训练权重 tag (CLIP_BACKEND=open_clip 时必填)
CLIP_PRETRAINED=""
```

参数区追加：

```bash
if [[ -z "$CLIP_BACKEND" ]]; then
  echo "Error: CLIP_BACKEND 必填 (openai_clip / open_clip / huggingface)。"
  exit 1
fi
ARGS+=(--clip_backend "$CLIP_BACKEND")
[[ -n "$CLIP_MODEL" ]]      && ARGS+=(--clip_model "$CLIP_MODEL")
[[ -n "$CLIP_PRETRAINED" ]] && ARGS+=(--clip_pretrained "$CLIP_PRETRAINED")
```

`CLIP_BACKEND` 必须明确指定其中一个值，shell 脚本和 python 都会拒绝空值。`CLIP_MODEL` 留空 = 沿用 YAML 里的 `model.vision_encoder`；填上即覆盖。

## 烟测结果

在 `copulasyn` 环境下验证三种 backbone 都能正常出 feature：

| Model | Pretrained | Backend | Feature dim |
| --- | --- | --- | --- |
| `ViT-B/16` | — | openai_clip | 512 |
| `RN50` | — | openai_clip | 1024 |
| `openai/clip-vit-base-patch16` | — | huggingface | 512 |
| `ViT-B-32` | `openai` | open_clip | 512 |

## 用法

### 方式 1: 编辑 shell 脚本顶部

```bash
# OpenAI clip 路径 (默认)
CLIP_BACKEND="openai_clip"
CLIP_MODEL="ViT-L/14"
CLIP_PRETRAINED=""

# open_clip 路径 (LAION-2B 训的 ViT-L)
CLIP_BACKEND="open_clip"
CLIP_MODEL="ViT-L-14"                  # 注意横线
CLIP_PRETRAINED="laion2b_s32b_b82k"

# HF 路径
CLIP_BACKEND="huggingface"
CLIP_MODEL="openai/clip-vit-large-patch14"
CLIP_PRETRAINED=""

bash script/evaluate/evaluate_cub2011_painting.sh
```

### 方式 2: 直接调 python

```bash
conda activate copulasyn

# OpenAI clip
python script/evaluate/evaluate_synthetic.py \
    --syn_index /path/to/syn_train_index.json \
    --config config/config_cub2011_painting.yaml \
    --clip_backend openai_clip \
    --clip_model ViT-L/14@336px

# open_clip with LAION-2B weights
python script/evaluate/evaluate_synthetic.py \
    --syn_index /path/to/syn_train_index.json \
    --config config/config_cub2011_painting.yaml \
    --clip_backend open_clip \
    --clip_model ViT-L-14 \
    --clip_pretrained laion2b_s32b_b82k

# HuggingFace
python script/evaluate/evaluate_synthetic.py \
    --syn_index /path/to/syn_train_index.json \
    --config config/config_cub2011_painting.yaml \
    --clip_backend huggingface \
    --clip_model openai/clip-vit-large-patch14
```

查看 open_clip 所有可用的 (架构, pretrained) 组合：

```bash
python -c "import open_clip; [print(p) for p in open_clip.list_pretrained()]"
```

## 注意

- 换 backbone 后第一次会触发该模型权重下载：
  - OpenAI clip: 缓存在 `~/.cache/clip/`（如 ViT-L/14 约 890 MB）
  - open_clip: 缓存在 `~/.cache/huggingface/hub/`（从 HF Hub 拉）
  - HuggingFace: 同上
- 之前烟测时 `~/.cache/clip/ViT-B-16.pt` 因下载中断 SHA256 校验失败，删掉后重下即可。
- 不同 backbone / pretrained 的 test feature 缓存彼此**不会**互相污染（已通过 backbone + pretrained slug 隔离）。
- HF 标识也可以填到 `CLIP_MODEL` 里，例如 `openai/clip-vit-large-patch14` — 会走 HF 路径；三路径都 work。
- 用 open_clip 加载 OpenAI 权重（`pretrained='openai'`）会有 QuickGELU 警告。如果一定要走 open_clip 用 openai 权重，应改用 `ViT-B-32-quickgelu` 这类带 `-quickgelu` 后缀的架构名；但更推荐用 OpenAI clip 路径直接加载（`CLIP_MODEL="ViT-B/32"`, `CLIP_PRETRAINED=""`）。

## 变更文件清单

```
src/evaluator/feature_extractor.py              三 backend 支持 (open_clip + openai_clip + huggingface)
src/evaluator/clip_evaluator.py                 zero-shot 改用 extractor API
script/evaluate/evaluate_synthetic.py           +--clip_model, +--clip_pretrained, cache 路径加 backbone+pretrained slug
script/evaluate/evaluate_synthetic.sh           +CLIP_MODEL, +CLIP_PRETRAINED
script/evaluate/evaluate_cub2011_painting.sh    +CLIP_MODEL, +CLIP_PRETRAINED
script/evaluate/evaluate_fgvc_aircraft.sh       +CLIP_MODEL, +CLIP_PRETRAINED
script/evaluate/evaluate_oxford_flowers102.sh   +CLIP_MODEL, +CLIP_PRETRAINED
script/evaluate/evaluate_stanford_car.sh        +CLIP_MODEL, +CLIP_PRETRAINED
```
