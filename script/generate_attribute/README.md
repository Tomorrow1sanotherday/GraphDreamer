# Attribute Generator Script

This script generates distinguishing attributes for subjects using LLM.

## Overview

The script reads co-objects from an input JSON file and generates distinguishing attributes that help differentiate each subject from other members of the same superclass.

## Usage

```bash
python run_generate_attribute.py \
    --input_path /path/to/coobjects.json \
    --output_dir /path/to/output_dir \
    --n_attributes 5 \
    --superclass Bird
```

## Arguments

- `--input_path` (required): Path to the input co-objects JSON file
- `--output_dir` (required): Directory to save the output JSON file
- `--n_attributes` (default: 5): Number of distinguishing attributes to generate per subject
- `--model_name` (default: deepseek-v3): Name of the LLM model to use
- `--max_concurrent` (default: 5): Maximum concurrent requests per API key
- `--config_path` (default: config/key.yaml): Path to the API keys config file
- `--superclass` (default: Bird): The broad category for all subjects
- `--max_retries_per_item` (default: 100): Maximum number of retries per item

## Input Format

The input JSON file should contain an array of objects with the following structure:

```json
[
  {
    "label": 0,
    "label_name": "Black_footed_Albatross",
    "subject": "Black footed Albatross",
    "semantically_associated": [...],
    "compatible_non_typical": [...],
    "contextually_contrastive": [...]
  },
  ...
]
```

## Output Format

The output JSON file will have the same structure as the input, with an additional `attributes` field:

```json
[
  {
    "label": 0,
    "label_name": "Black_footed_Albatross",
    "subject": "Black footed Albatross",
    "semantically_associated": [...],
    "compatible_non_typical": [...],
    "contextually_contrastive": [...],
    "attributes": [
      {"mount type": "wall"},
      {"color pattern": "black and white"},
      {"size": "large"},
      {"beak shape": "curved"},
      {"habitat": "ocean"}
    ]
  },
  ...
]
```

## Example

```bash
python run_generate_attribute.py \
    --input_path data/cub2011/co_objects/coobjects_20260103_001613.json \
    --output_dir data/cub2011/co_objects \
    --n_attributes 5 \
    --superclass Bird \
    --max_concurrent 10
```

## Notes

- Results are written to the output file in real-time as they are generated
- The script uses async processing for efficient parallel generation
- Each attribute is a dictionary with a single key-value pair
- The script will retry failed requests automatically

