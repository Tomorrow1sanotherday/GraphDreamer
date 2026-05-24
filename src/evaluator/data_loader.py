"""
Dataset loaders for synthetic and real image data.

Provides unified interfaces for loading:
- Synthetic data from JSON index files
- Synthetic data from ImageFolder-style directories (no JSON needed)
- Real data from HuggingFace datasets
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import re
from typing import Optional, Tuple, List, Any

import numpy as np
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset


def _strip_numeric_prefix(name: str) -> str:
    """Remove a leading numeric prefix like '001.' when present."""
    text = str(name).strip()
    if "." not in text:
        return text

    prefix, remainder = text.split(".", 1)
    return remainder if prefix.isdigit() else text


def _normalize_class_name(name: str) -> str:
    """Normalize class names for robust cross-dataset folder/name matching."""
    normalized = _strip_numeric_prefix(name).casefold()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def _ensure_rgb_image(image: Any) -> Image.Image:
    """Convert common dataset image payloads to RGB PIL images."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, dict):
        if "path" in image and image["path"]:
            return Image.open(image["path"]).convert("RGB")
        if "bytes" in image and image["bytes"]:
            from io import BytesIO

            return Image.open(BytesIO(image["bytes"])).convert("RGB")

    raise TypeError(f"Unsupported image payload type: {type(image)!r}")


class SyntheticDataset(Dataset):
    """
    Dataset for synthetic images generated from scene graphs.
    
    Loads images based on a JSON index file containing metadata
    including image paths, labels, and scene graph information.
    """
    
    def __init__(
        self,
        index_json_path: str,
        transform: Optional[Any] = None,
        max_samples: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            index_json_path: Path to the synthetic data index JSON file
            transform: Optional transform to apply to images
            max_samples: Maximum number of samples to use (None = use all).
                        Samples are drawn uniformly from each class.
            random_state: Random seed for reproducibility when sampling
        """
        self.index_path = Path(index_json_path)
        self.transform = transform
        
        with open(self.index_path, 'r') as f:
            data = json.load(f)
        
        self.metadata = {
            'dataset': data.get('dataset', 'unknown'),
            'total': data.get('total', 0),
            'image_dir': data.get('image_dir', '')
        }
        
        all_samples = data['results']
        self.samples = self._sample_uniformly(all_samples, max_samples, random_state)
        self.label_names = self._build_label_mapping()
    
    def _sample_uniformly(
        self,
        samples: List[dict],
        max_samples: Optional[int],
        random_state: int
    ) -> List[dict]:
        """Sample uniformly from each class if max_samples is specified."""
        if max_samples is None or max_samples >= len(samples):
            return samples
        
        np.random.seed(random_state)
        
        # Group samples by class
        class_samples = defaultdict(list)
        for sample in samples:
            class_samples[sample['label']].append(sample)
        
        num_classes = len(class_samples)
        samples_per_class = max_samples // num_classes
        remainder = max_samples % num_classes
        
        # Sample from each class
        selected = []
        for i, label in enumerate(sorted(class_samples.keys())):
            pool = class_samples[label]
            n_to_sample = samples_per_class + (1 if i < remainder else 0)
            n_to_sample = min(n_to_sample, len(pool))
            
            if n_to_sample < len(pool):
                indices = np.random.choice(len(pool), n_to_sample, replace=False)
                selected.extend([pool[idx] for idx in indices])
            else:
                selected.extend(pool)
        
        print(f"Sampled {len(selected)} samples uniformly from {num_classes} classes "
              f"(~{samples_per_class} per class)")
        return selected
    
    def _build_label_mapping(self) -> dict:
        """Build mapping from label index to label name."""
        mapping = {}
        for sample in self.samples:
            mapping[sample['label']] = sample['label_name']
        return mapping
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, int, str]:
        """
        Get a sample.
        
        Returns:
            Tuple of (image, label, label_name)
            - image: PIL Image or transformed tensor
            - label: Integer class label
            - label_name: String class name
        """
        sample = self.samples[idx]
        
        image = Image.open(sample['image_path']).convert('RGB')
        label = sample['label']
        label_name = sample['label_name']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, label_name
    
    def get_num_classes(self) -> int:
        """Return the number of unique classes."""
        return len(self.label_names)
    
    def get_class_names(self) -> List[str]:
        """Return list of class names sorted by label index."""
        max_label = max(self.label_names.keys())
        return [self.label_names.get(i, f"class_{i}") for i in range(max_label + 1)]


class SyntheticDirDataset(Dataset):
    """
    Dataset for synthetic images organized as ImageFolder (class_name/image.jpg).

    Unlike SyntheticDataset (which requires a JSON index), this class reads
    directly from a directory where each subdirectory is a class.

    Labels can be remapped from ImageFolder alphabetical order to a reference
    dataset order so they match the real evaluation split labels. Pass
    ``target_class_names`` to enable remapping; omit it to use raw 0-based
    alphabetical labels.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Any] = None,
        max_samples: Optional[int] = None,
        random_state: int = 42,
        target_class_names: Optional[List[str]] = None,
        cub_class_names: Optional[List[str]] = None,
    ):
        """
        Initialize directory-based synthetic dataset.

        Args:
            root: Root directory containing per-class subdirectories.
            transform: Optional transform to apply to images.
            max_samples: Maximum number of samples (None = all).
                         Samples are drawn uniformly from each class.
            random_state: Random seed for reproducibility when sampling.
            target_class_names: Reference class names in target label order.
                Used to remap ImageFolder alphabetical labels to the reference
                dataset indices. If None, labels are left as 0-based
                alphabetical indices.
            cub_class_names: Backward-compatible alias for target_class_names.
        """
        from torchvision.datasets import ImageFolder

        self.root = Path(root)
        self.transform = transform

        if target_class_names is not None and cub_class_names is not None:
            raise ValueError("Use either target_class_names or cub_class_names, not both.")
        if target_class_names is None:
            target_class_names = cub_class_names

        folder_ds = ImageFolder(str(self.root))

        # Build label remap: folder_idx -> target_label
        if target_class_names is not None:
            target_norm_to_idx = {
                _normalize_class_name(name): i for i, name in enumerate(target_class_names)
            }
            folder_to_target: List[Optional[int]] = []
            missing = []
            for folder_name in folder_ds.classes:
                idx = target_norm_to_idx.get(_normalize_class_name(folder_name))
                if idx is None:
                    missing.append(folder_name)
                folder_to_target.append(idx)
            if missing:
                raise ValueError(
                    f"Following synthetic class folders could not be matched to "
                    f"reference label names: {missing[:10]}"
                    + (" ..." if len(missing) > 10 else "")
                )
        else:
            folder_to_target = list(range(len(folder_ds.classes)))

        all_samples = [
            {
                "image_path": img_path,
                "label": folder_to_target[folder_idx],
                "label_name": folder_ds.classes[folder_idx],
            }
            for img_path, folder_idx in folder_ds.samples
        ]

        self.samples = self._sample_uniformly(all_samples, max_samples, random_state)
        self.label_names = {s["label"]: s["label_name"] for s in self.samples}

    def _sample_uniformly(
        self,
        samples: List[dict],
        max_samples: Optional[int],
        random_state: int,
    ) -> List[dict]:
        """Sample uniformly from each class if max_samples is specified."""
        if max_samples is None or max_samples >= len(samples):
            return samples

        np.random.seed(random_state)

        class_samples: dict = defaultdict(list)
        for s in samples:
            class_samples[s["label"]].append(s)

        num_classes = len(class_samples)
        per_class = max_samples // num_classes
        remainder = max_samples % num_classes

        selected: List[dict] = []
        for i, label in enumerate(sorted(class_samples.keys())):
            pool = class_samples[label]
            n = per_class + (1 if i < remainder else 0)
            n = min(n, len(pool))
            if n < len(pool):
                indices = np.random.choice(len(pool), n, replace=False)
                selected.extend([pool[idx] for idx in indices])
            else:
                selected.extend(pool)

        print(
            f"Sampled {len(selected)} samples uniformly from {num_classes} classes "
            f"(~{per_class} per class)"
        )
        return selected

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int, str]:
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, s["label"], s["label_name"]

    def get_num_classes(self) -> int:
        return len(self.label_names)

    def get_class_names(self) -> List[str]:
        max_label = max(self.label_names.keys())
        return [self.label_names.get(i, f"class_{i}") for i in range(max_label + 1)]


class RealDataset(Dataset):
    """
    Dataset wrapper for real images from HuggingFace datasets.
    
    Provides a consistent interface matching SyntheticDataset.
    """
    
    def __init__(
        self,
        split: str = 'train',
        transform: Optional[Any] = None,
        dataset_path: Optional[str] = None,
        dataset_config: Optional[dict] = None,
    ):
        """
        Initialize real dataset.
        
        Args:
            split: Dataset split ('train' or 'test')
            transform: Optional transform to apply to images
            dataset_path: Path to HuggingFace dataset cache
            dataset_config: Generic dataset configuration. Supports:
                - loader: 'huggingface' or 'imagefolder'
                - path: dataset path
                - splits: mapping from logical split names to actual names
                - image_column: image column for HuggingFace datasets
                - label_column: label column for HuggingFace datasets
                - label_name_column: optional class-name column
                - class_names: optional explicit class-name list
                - strip_label_prefix: remove prefixes like '001.' from names
        """
        self.dataset_config = dict(dataset_config or {})
        self.loader_type = self.dataset_config.get('loader', 'huggingface')
        self.dataset_name = self.dataset_config.get('name', 'dataset')
        self.image_column = self.dataset_config.get('image_column', 'image')
        self.label_column = self.dataset_config.get('label_column', 'label')
        self.label_name_column = self.dataset_config.get('label_name_column')
        self.strip_label_prefix = self.dataset_config.get('strip_label_prefix', False)

        split_map = self.dataset_config.get('splits', {})
        split_name = split_map.get(split, split)

        if dataset_path is None:
            dataset_path = self.dataset_config.get('path')
        if dataset_path is None:
            raise ValueError("dataset_path or dataset_config['path'] must be provided.")

        self.dataset_path = dataset_path
        self.transform = transform
        self.split = split

        if self.loader_type == 'huggingface':
            self.dataset = load_dataset(dataset_path, split=split_name)
            self._raw_label_to_index = None
            self._label_names = self._build_huggingface_label_names()
        elif self.loader_type == 'imagefolder':
            from torchvision.datasets import ImageFolder

            split_root = self._resolve_imagefolder_split_root(dataset_path, split_name)
            self.dataset = ImageFolder(str(split_root))
            self._raw_label_to_index = None
            self._label_names = [self._format_class_name(name) for name in self.dataset.classes]
        else:
            raise ValueError(
                f"Unsupported dataset loader '{self.loader_type}'. "
                f"Expected 'huggingface' or 'imagefolder'."
            )

    def _format_class_name(self, name: Any) -> str:
        text = str(name)
        return _strip_numeric_prefix(text) if self.strip_label_prefix else text

    def _resolve_imagefolder_split_root(self, dataset_path: str, split_name: str) -> Path:
        root = Path(dataset_path)
        split_root = root / split_name
        if split_name in ('', '.', None):
            return root
        return split_root if split_root.exists() else root

    def _build_huggingface_label_names(self) -> List[str]:
        class_names = self.dataset_config.get('class_names')
        if class_names is not None:
            return [self._format_class_name(name) for name in class_names]

        label_feature = self.dataset.features.get(self.label_column)
        if hasattr(label_feature, 'names') and label_feature.names is not None:
            return [self._format_class_name(name) for name in label_feature.names]

        raw_labels = list(self.dataset[self.label_column])

        if self.label_name_column:
            raw_label_names = list(self.dataset[self.label_name_column])
            label_to_name = {}
            for raw_label, raw_name in zip(raw_labels, raw_label_names):
                if raw_label not in label_to_name:
                    label_to_name[raw_label] = self._format_class_name(raw_name)

            first_label = raw_labels[0] if raw_labels else None
            if isinstance(first_label, (int, np.integer)):
                max_label = max(int(label) for label in label_to_name.keys()) if label_to_name else -1
                return [label_to_name.get(i, f'class_{i}') for i in range(max_label + 1)]

            ordered_labels = list(label_to_name.keys())
            self._raw_label_to_index = {
                raw_label: index for index, raw_label in enumerate(ordered_labels)
            }
            return [label_to_name[raw_label] for raw_label in ordered_labels]

        first_label = raw_labels[0] if raw_labels else None
        if isinstance(first_label, (int, np.integer)):
            max_label = max(int(label) for label in raw_labels) if raw_labels else -1
            return [f'class_{i}' for i in range(max_label + 1)]

        ordered_labels = list(dict.fromkeys(raw_labels))
        self._raw_label_to_index = {
            raw_label: index for index, raw_label in enumerate(ordered_labels)
        }
        return [self._format_class_name(raw_label) for raw_label in ordered_labels]
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, int, str]:
        """
        Get a sample.
        
        Returns:
            Tuple of (image, label, label_name)
        """
        if self.loader_type == 'imagefolder':
            image, label = self.dataset[idx]
            image = image.convert('RGB')
            label_name = self._label_names[int(label)]
        else:
            item = self.dataset[idx]
            image = _ensure_rgb_image(item[self.image_column])
            raw_label = item[self.label_column]
            if self._raw_label_to_index is None:
                label = int(raw_label)
            else:
                label = self._raw_label_to_index[raw_label]
            label_name = self._label_names[label]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, label_name
    
    def get_num_classes(self) -> int:
        """Return the number of unique classes."""
        return len(self._label_names)
    
    def get_class_names(self) -> List[str]:
        """Return list of class names sorted by label index."""
        # Strip a leading "NNN." numeric prefix only; preserve dots that appear
        # inside the actual class name (e.g. "Bugatti Veyron 16.4 Coupe 2009").
        return [_strip_numeric_prefix(name) for name in self._label_names]

