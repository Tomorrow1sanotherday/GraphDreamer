# DivSyn

**Setup:** Python 3.10+, PyTorch, transformers, diffusers, scikit-learn, PyYAML, tqdm. Put LLM API keys in `config/key.yaml`. Edit `config/config.yaml` for dataset and paths.

Real dataset loading is configuration-driven now.

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

**1. Co-objects**
```bash
python script/generate_coobject/run_generate_coobject.py --input_path labels.json --output_dir out --n_objects_per_category 20
```

**2. Attributes**
```bash
python script/generate_attribute/run_generate_attribute.py --input_path coobjects.json --output_dir out --n_attributes 5 --superclass Bird
```

**3. Scene graphs**
```bash
python script/generate_scene_graph/run_generate_scene_graph.py --input_path coobjects.json --output_path scene_graphs.json --samples_per_subject 10
```

**4. Images**
```bash
python script/generate_image/run_generate_image.py --input_path scene_graphs.json --output_dir images --output_json out.json
```

**5. Evaluation**
```bash
python script/evaluate/evaluate_synthetic.py --syn_index syn_train_index.json --n_per_class 3 --results_dir results
```

Run from repo root. Use `--help` on each script for more options.
