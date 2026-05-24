# GraphDreamer

Diversity- and realism-aware synthetic data generation for fine-grained image classification, built around scene-graph construction, LLM-based realism scoring, and greedy submodular subset selection.

**Setup:** Python 3.10+, PyTorch, transformers, diffusers, scikit-learn, PyYAML, tqdm. Put LLM API keys in `config/key.yaml`. Edit `config/config.yaml` (or one of the `config/config_<dataset>.yaml` files) for dataset and paths.

## Dataset configuration

Real dataset loading is configuration-driven.

HuggingFace dataset example in `config/config.yaml`:
```yaml
dataset:
	name: "cub_200_2011"
	loader: "huggingface"
	path: "/path/to/huggingface/dataset"
	splits:
		train: "train"
		test: "test"
	image_column: "image"
	label_column: "label"
	label_name_column: null
	strip_label_prefix: true
	cache:
		test_features_path: "./features/cub_200_2011_real_test_features.npz"
```

ImageFolder dataset example for datasets such as CUB-200-Painting:
```yaml
dataset:
	name: "cub_200_painting"
	loader: "imagefolder"
	path: "/path/to/CUB-200-Painting"
	splits:
		test: "."
	strip_label_prefix: true
	cache:
		test_features_path: "./features/cub_200_painting_real_test_features.npz"
```

For `imagefolder`, the evaluation split should be either:
- a root directory with `class_name/image.jpg` and `test: "."`
- or a split subdirectory like `test/class_name/image.jpg` and `test: "test"`

## Pipeline

The full pipeline goes from class labels to an evaluation-ready synthetic subset that has been pruned for both structural diversity and realism. Stage 2 is optional; stages 5–8 form the optimization stack and can be skipped if you only want raw synthesis.

### 1. Co-objects
LLM-generated co-occurring objects per class label.
```bash
python script/generate_coobject/run_generate_coobject.py \
	--input_path labels.json --output_dir out \
	--n_objects_per_category 20 --superclass Bird
```

### 2. Attributes (optional)
LLM-generated subject attributes (pose, activity, condition) per class — fed into scene graph construction as `scene_graph.attributes.state`. The script is named `generate_state` for historical reasons; the output is what the rest of the pipeline calls *attributes*.
```bash
python script/generate_state/run_generate_state.py \
	--input_path labels.json --output_dir out \
	--n_states 20 --superclass Bird
```

### 3. Scene graphs
Programmatic scene graph construction (no LLM) from co-objects, a per-dataset relation vocabulary (curated as a flat JSON array at `data/<dataset>/ralations.json`, e.g. `["on", "near", "above", "in", "next to"]`), and the attributes JSON from stage 2 (passed via `--states_path`). Style / painting axes are optional.
```bash
python script/generate_scene_graph/run_generate_sg_programmtic.py \
	--coobjects_path coobjects_5.json \
	--relations_path data/<dataset>/ralations.json \
	--states_path attributes.json \
	--output_path scene_graphs.json \
	--samples_per_subject 10 --min_objects 1 --max_objects 2
```
Disable individual axes with `--no-state` / `--no-relation` / `--no-coobject` / `--no-style` / `--no-painting`.

### 4. Images
Multi-GPU diffusion image generation from scene graphs.
```bash
python script/generate_image/run_generate_image.py \
	--input_path scene_graphs.json \
	--output_dir images --output_json syn_train_index.json
```
A base-prompt variant (AttrSyn-style flat prompts) is available at
`script/generate_image/generate_base_image/run_generate_base_image.py`.

### 5. Reality QA
Build feasibility-checking questions (`R_SA`, `R_SO`, `R_SRO`, `R_SCENE`) for each scene graph.
```bash
python script/generate_qa/run_generate_reality_qa.py \
	--input_path scene_graphs.json --output_dir out
```

### 6. Score realism
LLM zero-shot scores each question in `[0, 1]`; `R(g)` is the mean per scene graph.
```bash
python script/optimize_scene_graph/run_score_realism.py \
	--input_path reality_qa.json \
	--output_path realism_scores.json \
	--model_name deepseek-v4-flash
```

### 7. Optimize scene graphs
Greedy submodular selection of `K` scene graphs per class that maximize
`λ · D(G*) + (1−λ) · R̄(G*)`, where `D` is structural-entropy diversity and `R` is realism.
```bash
python script/optimize_scene_graph/run_optimize_sg.py \
	--input_path scene_graphs_with_realism.json \
	--output_path scene_graphs_optimized.json \
	--budget_per_class 30 --lambda_weight 0.5
```

### 8. Build optimized eval subset
Materialize the optimizer's selection into a `syn_train_index`-style JSON plus a per-class image directory, ready for evaluation.
```bash
python script/optimize_scene_graph/build_optimized_eval_subset.py \
	--optimized_json scene_graphs_optimized.json \
	--source_index_json syn_train_index.json \
	--output_json syn_train_index_optimized.json \
	--output_image_dir syn_images_optimized
```

### 9. Evaluation
Train CLIP classifiers (LR / MLP) on the synthetic subset and evaluate against the real test split defined by `config.yaml`.
```bash
python script/evaluate/evaluate_synthetic.py \
	--syn_index syn_train_index_optimized.json \
	--n_per_class 3 --results_dir results
```

## Repository layout

```
src/
	api/         async LLM client + streaming generator
	generators/  co-object / attribute (state_generator/) / scene-graph / image / QA generators
	metrics/     diversity, realism, greedy selection (used by stage 8)
	evaluator/   CLIP feature extractor, classifier trainer, zero-shot evaluator
script/        thin CLI wrappers around src/ — one folder per pipeline stage
config/        dataset configs + key.yaml (gitignored)
```

Run all scripts from the repo root. Use `--help` on each for the full flag list. Convenience `*.sh` wrappers live alongside the `*.py` entry points for common dataset presets.
