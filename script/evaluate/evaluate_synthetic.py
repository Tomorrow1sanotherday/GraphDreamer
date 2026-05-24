#!/usr/bin/env python3
"""
Synthetic Data Evaluation Pipeline

Evaluates synthetic data quality by comparing:
1. CLIP Zero-Shot classification (baseline, no training)
2. CLIP + Logistic Regression trained on synthetic data
3. CLIP + Logistic Regression trained on real data (fair comparison)
4. CLIP + Logistic Regression trained on synthetic + real data

Input modes (mutually exclusive, one required):
  --syn_index PATH     JSON index file produced by the DivSyn generator.
  --synthetic_root DIR Directory of images in ImageFolder format
                       (class_name/image.jpg), no JSON needed.

Usage:
    # JSON index mode (original)
    python evaluate_synthetic.py --syn_index path/to/syn_train_index.json

    # Directory mode (no JSON)
    python evaluate_synthetic.py --synthetic_root path/to/syn_images/

Arguments:
    --syn_index         Path to synthetic data index JSON file
    --synthetic_root    Path to synthetic data directory (ImageFolder format)
    --n_per_class       Number of real samples per class for fair comparison (default: 3)
    --syn_samples       Limit synthetic samples (default: all)
    --batch_size        Batch size for feature extraction (default: 32)
    --no_zero_shot      Skip zero-shot evaluation
    --results_dir       Directory to save results (default: ./results)
    --detailed_results  Save per-sample predictions (true vs pred class) for analysis
"""

import sys
import os

# Get project root (script is in script/evaluate/, so go up two levels)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import argparse
import csv
import re
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torchvision.transforms import Resize

from src.evaluator import (
    CLIPFeatureExtractor,
    CLIPClassifierTrainer,
    CLIPMLPClassifierTrainer,
    CLIPZeroShotEvaluator,
    SyntheticDataset,
    SyntheticDirDataset,
    RealDataset,
    EvaluationResultsManager,
)


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate synthetic data quality using CLIP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input source: exactly one of the two must be supplied
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-s',
        '--syn_index',
        type=str,
        default=None,
        help='Path to synthetic data index JSON file (DivSyn generator output).'
    )
    group.add_argument(
        '--synthetic_root',
        type=str,
        default=None,
        help='Path to synthetic data directory in ImageFolder format '
             '(class_name/image.jpg). No JSON index needed.'
    )
    
    # Training parameters
    parser.add_argument(
        '--n_per_class',
        type=int,
        default=3,
        help='Number of real samples per class for fair comparison (default: 3)'
    )
    parser.add_argument(
        '--syn_samples',
        type=int,
        default=None,
        help='Maximum number of synthetic samples to use (default: all)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for feature extraction (default: 32)'
    )
    
    # Evaluation options
    parser.add_argument(
        '--no_zero_shot',
        action='store_true',
        help='Skip CLIP zero-shot evaluation'
    )
    
    # Output options
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        help='Directory to save results (default: ./results)'
    )
    parser.add_argument(
        '--features_dir',
        type=str,
        default='./features',
        help='Directory to cache extracted features (default: ./features)'
    )
    parser.add_argument(
        '--detailed_results',
        action='store_true',
        help='Save per-sample predictions (true vs pred class) to CSV for confusion analysis'
    )
    
    parser.add_argument(
        '--resize',
        type=int,
        default=None,
        help='Resize images to (resize, resize) before feature extraction. If not set, use original size.'
    )
    
    # Other options
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to dataset config YAML (default: config/config.yaml)'
    )

    parser.add_argument(
        '--clip_backend',
        type=str,
        required=True,
        choices=['openai_clip', 'open_clip', 'huggingface'],
        help=(
            'CLIP backend to use (REQUIRED, no auto-detection). '
            '"openai_clip" → official OpenAI clip package, names like "ViT-L/14". '
            '"open_clip" → LAION/DataComp etc., requires --clip_pretrained, '
            'arch names use dashes like "ViT-L-14". '
            '"huggingface" → transformers.CLIPModel, names like '
            '"openai/clip-vit-large-patch14".'
        )
    )

    parser.add_argument(
        '--clip_model',
        type=str,
        default=None,
        help=(
            'Override config.model.vision_encoder. Format depends on --clip_backend: '
            'openai_clip → "RN50"/"ViT-L/14"/...; '
            'open_clip → "ViT-L-14"/"ViT-H-14"/...; '
            'huggingface → "openai/clip-vit-large-patch14"/etc.'
        )
    )

    parser.add_argument(
        '--clip_pretrained',
        type=str,
        default=None,
        help=(
            'open_clip pretrained tag (REQUIRED when --clip_backend=open_clip). '
            'Examples: "laion2b_s32b_b82k", "datacomp_xl_s13b_b90k". '
            'Run `python -c "import open_clip; print(open_clip.list_pretrained())"` '
            'for the full list.'
        )
    )

    return parser.parse_args()


def _extract_all_labels(dataset: Dataset) -> np.ndarray:
    """Extract all labels from a dataset without forcing image decoding when possible."""
    if hasattr(dataset, 'samples'):
        return np.asarray([int(sample['label']) for sample in dataset.samples], dtype=np.int64)

    base_dataset = getattr(dataset, 'dataset', None)
    if base_dataset is not None and hasattr(base_dataset, 'column_names') and 'label' in base_dataset.column_names:
        return np.asarray(base_dataset['label'], dtype=np.int64)

    labels = []
    for idx in range(len(dataset)):
        _, label, _ = dataset[idx]
        labels.append(int(label))
    return np.asarray(labels, dtype=np.int64)


class RemappedClassSubsetDataset(Dataset):
    """Filter a dataset to a class subset and remap labels to a contiguous range."""

    def __init__(self, base_dataset: Dataset, original_labels: Iterable[int], name: str):
        self.base_dataset = base_dataset
        self.name = name
        self.original_labels = sorted({int(label) for label in original_labels})
        if not self.original_labels:
            raise ValueError(f"No class labels provided for {name} subset.")

        self.original_to_new = {
            original_label: new_label
            for new_label, original_label in enumerate(self.original_labels)
        }

        self.base_labels = _extract_all_labels(base_dataset)
        self.original_indices = np.flatnonzero(np.isin(self.base_labels, self.original_labels))
        if len(self.original_indices) == 0:
            raise ValueError(f"No samples from the requested classes were found in {name}.")

        self.class_names = self._build_class_names()

    def _build_class_names(self) -> List[str]:
        if hasattr(self.base_dataset, 'get_class_names'):
            all_class_names = self.base_dataset.get_class_names()
            return [
                all_class_names[label] if 0 <= label < len(all_class_names) else f"class_{label}"
                for label in self.original_labels
            ]

        label_to_name = {}
        for base_idx in self.original_indices:
            _, label, label_name = self.base_dataset[int(base_idx)]
            label_to_name[int(label)] = label_name
            if len(label_to_name) == len(self.original_labels):
                break

        return [label_to_name.get(label, f"class_{label}") for label in self.original_labels]

    def __len__(self) -> int:
        return len(self.original_indices)

    def __getitem__(self, idx: int):
        base_idx = int(self.original_indices[idx])
        image, original_label, _ = self.base_dataset[base_idx]
        remapped_label = self.original_to_new[int(original_label)]
        return image, remapped_label, self.class_names[remapped_label]

    def remap_labels(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels)
        return np.asarray([self.original_to_new[int(label)] for label in labels], dtype=np.int64)

    def subset_cached_features(self, features: np.ndarray, labels: np.ndarray):
        labels = np.asarray(labels)
        if len(labels) != len(self.base_labels):
            raise ValueError(
                f"Cached labels length {len(labels)} does not match base dataset length "
                f"{len(self.base_labels)} for {self.name}."
            )

        subset_features = features[self.original_indices]
        subset_labels = self.remap_labels(labels[self.original_indices])
        return subset_features, subset_labels

    def get_num_classes(self) -> int:
        return len(self.class_names)

    def get_class_names(self) -> List[str]:
        return list(self.class_names)

    def get_original_labels(self) -> List[int]:
        return list(self.original_labels)


class SyntheticDataEvaluator:
    """
    Main evaluator class for synthetic data quality assessment.

    Supports two input modes:
    - JSON index (syn_index_path): original DivSyn format
    - Directory (synthetic_root): ImageFolder format, no JSON needed
    """

    def __init__(
        self,
        config: dict,
        syn_index_path: str = None,
        synthetic_root: str = None,
        n_per_class: int = 3,
        syn_samples: int = None,
        batch_size: int = 32,
        results_dir: str = './results',
        features_dir: str = './features',
        seed: int = 42,
        detailed_results: bool = False,
        resize: int = None
    ):
        """
        Initialize evaluator.

        Args:
            config: Configuration dictionary
            syn_index_path: Path to synthetic data index JSON (mutually exclusive
                with synthetic_root)
            synthetic_root: Path to synthetic data directory in ImageFolder format
                (mutually exclusive with syn_index_path)
            n_per_class: Number of real samples per class for fair comparison
            syn_samples: Maximum synthetic samples (None = all)
            batch_size: Batch size for processing
            results_dir: Directory for results
            features_dir: Directory for cached features
            seed: Random seed
            detailed_results: If True, save per-sample predictions to CSV for analysis
            resize: If set, resize images to (resize, resize) before feature extraction
        """
        if not syn_index_path and not synthetic_root:
            raise ValueError("One of syn_index_path or synthetic_root must be provided.")
        if syn_index_path and synthetic_root:
            raise ValueError("syn_index_path and synthetic_root are mutually exclusive.")

        self.config = config
        self.dataset_config = dict(config.get('dataset', {}))
        self.syn_index_path = syn_index_path
        self.synthetic_root = synthetic_root
        self.n_per_class = n_per_class
        self.syn_samples = syn_samples
        self.batch_size = batch_size
        self.seed = seed
        self.detailed_results = detailed_results
        self.resize = resize
        
        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(results_dir) / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.shared_features_dir = Path(features_dir)
        self.shared_features_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize paths (both in the same subdirectory)
        self.features_dir = self.run_dir / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results manager (results also in the same subdirectory)
        self.results = EvaluationResultsManager(results_dir=str(self.run_dir))
        
        # Shared cache for real test features across runs, scoped by dataset.
        self.test_features_path = self._resolve_test_features_cache_path()
        
        # Components (initialized lazily)
        self._extractor = None
        self._raw_datasets = {}
        self._datasets = {}
        self._features = {}
        self.eval_original_labels = []
        self.active_class_names = []

    def _slugify(self, value: str) -> str:
        text = str(value).strip().lower()
        text = re.sub(r'[^a-z0-9]+', '_', text)
        return text.strip('_') or 'dataset'

    def _resolve_test_features_cache_path(self) -> Path:
        cache_config = self.dataset_config.get('cache', {})
        configured_path = cache_config.get('test_features_path')
        model_cfg = self.config.get('model', {})
        backbone_slug = self._slugify(model_cfg.get('vision_encoder', 'clip'))
        pretrained = model_cfg.get('vision_encoder_pretrained')
        if pretrained:
            backbone_slug = f"{backbone_slug}__{self._slugify(pretrained)}"

        if configured_path:
            # Honour user-supplied cache path, but namespace by backbone so
            # cached HF-ViT-L features don't get reused for an OpenAI RN50 run.
            p = Path(configured_path)
            return p.with_name(f"{p.stem}__{backbone_slug}{p.suffix}")

        dataset_name = self.dataset_config.get('name', 'dataset')
        return self.shared_features_dir / f"{self._slugify(dataset_name)}__{backbone_slug}_real_test_features.npz"
    
    @property
    def extractor(self) -> CLIPFeatureExtractor:
        """Lazy initialization of CLIP feature extractor."""
        if self._extractor is None:
            model_cfg = self.config['model']
            model_name = model_cfg['vision_encoder']
            pretrained = model_cfg.get('vision_encoder_pretrained')
            backend = model_cfg.get('vision_encoder_backend')
            self._extractor = CLIPFeatureExtractor(
                model_name=model_name,
                pretrained=pretrained,
                backend=backend,
            )
        return self._extractor
    
    def load_datasets(self) -> None:
        """Load all required datasets."""
        print("\n" + "=" * 60)
        print("Loading Datasets")
        print("=" * 60)

        transform = Resize((self.resize, self.resize)) if self.resize is not None else None
        if transform is not None:
            print(f"Resize: {self.resize}x{self.resize}")

        dataset_name = self.dataset_config.get('name', 'dataset')
        dataset_path = self.dataset_config.get('path')

        if self.synthetic_root:
            # Directory mode: load real dataset first to get reference class names
            # for label remapping, then load synthetic with matching label indices.
            print(f"\nLoading real test data '{dataset_name}' from: {dataset_path}")
            self._raw_datasets['real_test'] = RealDataset(
                split='test',
                transform=transform,
                dataset_config=self.dataset_config,
            )
            print(f"Raw real test samples: {len(self._raw_datasets['real_test'])}")

            target_class_names = self._raw_datasets['real_test'].get_class_names()
            print(f"\nLoading synthetic data from directory: {self.synthetic_root}")
            self._raw_datasets['syn_train'] = SyntheticDirDataset(
                root=self.synthetic_root,
                transform=transform,
                max_samples=self.syn_samples,
                random_state=self.seed,
                target_class_names=target_class_names,
            )
        else:
            # JSON index mode (original behaviour)
            print(f"\nLoading synthetic data from: {self.syn_index_path}")
            self._raw_datasets['syn_train'] = SyntheticDataset(
                index_json_path=self.syn_index_path,
                transform=transform,
                max_samples=self.syn_samples,
                random_state=self.seed
            )
            print(f"Raw synthetic training samples: {len(self._raw_datasets['syn_train'])}")

            print(f"\nLoading real test data '{dataset_name}' from: {dataset_path}")
            self._raw_datasets['real_test'] = RealDataset(
                split='test',
                transform=transform,
                dataset_config=self.dataset_config,
            )
            print(f"Raw real test samples: {len(self._raw_datasets['real_test'])}")

        if self.synthetic_root:
            print(f"Raw synthetic training samples: {len(self._raw_datasets['syn_train'])}")

        train_original_labels = sorted(np.unique(_extract_all_labels(self._raw_datasets['syn_train'])).astype(int).tolist())
        test_original_labels = set(np.unique(_extract_all_labels(self._raw_datasets['real_test'])).astype(int).tolist())
        self.eval_original_labels = [label for label in train_original_labels if label in test_original_labels]

        if not self.eval_original_labels:
            raise ValueError(
                "No overlapping classes were found between synthetic training data and the real test split."
            )

        dropped_train_only_labels = [label for label in train_original_labels if label not in test_original_labels]
        if dropped_train_only_labels:
            preview = dropped_train_only_labels[:10]
            suffix = " ..." if len(dropped_train_only_labels) > 10 else ""
            print(
                f"Dropping {len(dropped_train_only_labels)} training-only classes with no real test samples: "
                f"{preview}{suffix}"
            )

        self._datasets['syn_train'] = RemappedClassSubsetDataset(
            self._raw_datasets['syn_train'],
            self.eval_original_labels,
            name='synthetic train'
        )
        self._datasets['real_test'] = RemappedClassSubsetDataset(
            self._raw_datasets['real_test'],
            self.eval_original_labels,
            name='real test'
        )
        self.active_class_names = self._datasets['real_test'].get_class_names()

        print(f"\nClasses inferred from synthetic training data: {len(train_original_labels)}")
        print(f"Evaluation classes used (train ∩ test): {len(self.eval_original_labels)}")
        print(f"Filtered synthetic training samples: {len(self._datasets['syn_train'])}")
        print(f"Filtered real test samples: {len(self._datasets['real_test'])}")
        preview_names = ", ".join(self.active_class_names[:10])
        preview_suffix = " ..." if len(self.active_class_names) > 10 else ""
        print(f"Active classes: {preview_names}{preview_suffix}")
    
    def extract_features(self) -> None:
        """Extract CLIP features from all datasets."""
        print("\n" + "=" * 60)
        print("Extracting CLIP Features")
        print("=" * 60)
        
        # Synthetic features
        print("\n[Synthetic Training Data]")
        syn_path = self.features_dir / "syn_train_features.npz"
        syn_features, syn_labels, _ = self.extractor.extract_dataset_features(
            self._datasets['syn_train'],
            batch_size=self.batch_size,
            save_path=str(syn_path)
        )
        self._features['syn_train'] = (syn_features, syn_labels)
        
        # Real test features: use fixed path only when no resize (same resolution as pre-computed)
        print("\n[Real Test Data]")
        if self.resize is not None:
            # Resize is set: always extract from dataset so test uses same resolution as train
            print(f"Resize={self.resize}: extracting test features from dataset (same resolution as train)")
            test_path = self.features_dir / "real_test_features.npz"
            test_features, test_labels, _ = self.extractor.extract_dataset_features(
                self._datasets['real_test'],
                batch_size=self.batch_size,
                save_path=str(test_path)
            )
        elif self.test_features_path.exists():
            print(f"Loading test features from: {self.test_features_path}")
            data = np.load(self.test_features_path, allow_pickle=True)
            cached_test_features = data['features']
            cached_test_labels = data['labels']
            print(f"Loaded cached full test features: {cached_test_features.shape}, labels: {cached_test_labels.shape}")

            raw_test_size = len(self._raw_datasets['real_test'])
            if len(cached_test_labels) != raw_test_size:
                print(
                    "Cached test features do not match the full real test split size; "
                    "extracting filtered test features from dataset instead..."
                )
                test_path = self.features_dir / "real_test_features.npz"
                test_features, test_labels, _ = self.extractor.extract_dataset_features(
                    self._datasets['real_test'],
                    batch_size=self.batch_size,
                    save_path=str(test_path)
                )
            else:
                test_features, test_labels = self._datasets['real_test'].subset_cached_features(
                    cached_test_features,
                    cached_test_labels,
                )
                print(f"Filtered cached test features to active classes: {test_features.shape}, labels: {test_labels.shape}")
        else:
            print(f"Test features not found at {self.test_features_path}, extracting from dataset...")
            test_path = self.features_dir / "real_test_features.npz"
            test_features, test_labels, _ = self.extractor.extract_dataset_features(
                self._datasets['real_test'],
                batch_size=self.batch_size,
                save_path=str(test_path)
            )
        
        self._features['real_test'] = (test_features, test_labels)
    
    def run_zero_shot(self) -> float:
        """Run CLIP zero-shot evaluation."""
        print("\n" + "=" * 60)
        print("CLIP Zero-Shot Evaluation")
        print("=" * 60)
        
        domain = self.dataset_config.get('domain')
        if not domain:
            raise ValueError(
                "dataset.domain is required in the config for zero-shot "
                "evaluation (used to build prompts like 'a {domain} of a "
                "{class_name}'). Add e.g. 'domain: photo' under 'dataset:'."
            )
        evaluator = CLIPZeroShotEvaluator(self.extractor, domain=domain)
        print(f"  Zero-shot prompt template: \"{evaluator.prompt_template}\"")
        accuracy, eval_results = evaluator.evaluate(
            self._datasets['real_test'],
            batch_size=self.batch_size
        )
        
        print(f"\n*** Zero-Shot Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) ***")
        
        # Get class names mapping
        class_names_mapping = self._get_class_names_mapping()
        
        # Calculate per-class accuracy with class names
        per_class_acc = self._calculate_per_class_accuracy(
            eval_results['predictions'],
            eval_results['labels'],
            class_names=class_names_mapping
        )
        
        self.results.add_result(
            name="CLIP Zero-Shot (no training)",
            accuracy=accuracy,
            num_train_samples=0,
            num_test_samples=len(self._datasets['real_test']),
            per_class_accuracy=per_class_acc
        )
        
        if self.detailed_results:
            self._save_detailed_predictions(
                eval_results['predictions'],
                eval_results['labels'],
                class_names_mapping,
                experiment_name="CLIP Zero-Shot",
                filename_prefix="detailed_predictions_zero_shot"
            )
        
        return accuracy
    
    def _calculate_per_class_accuracy(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray,
        class_names: dict = None
    ) -> dict:
        """
        Calculate accuracy for each class.
        
        Args:
            predictions: Predicted labels
            labels: Ground truth labels
            class_names: Dictionary mapping class_id to class_name (optional)
            
        Returns:
            Dictionary mapping class_id to dict with 'accuracy' and 'name' keys
        """
        per_class_acc = {}
        unique_classes = np.unique(labels)
        
        for class_id in unique_classes:
            class_mask = labels == class_id
            if np.sum(class_mask) > 0:
                class_predictions = predictions[class_mask]
                class_labels = labels[class_mask]
                class_acc = accuracy_score(class_labels, class_predictions)
                
                class_info = {
                    'accuracy': float(class_acc),
                    'name': class_names.get(int(class_id), f"class_{int(class_id)}") if class_names else f"class_{int(class_id)}"
                }
                per_class_acc[int(class_id)] = class_info
        
        return per_class_acc
    
    def _get_class_names_mapping(self) -> dict:
        """
        Get mapping from class_id to class_name from test dataset.
        
        Returns:
            Dictionary mapping class_id to class_name
        """
        test_dataset = self._datasets['real_test']
        if hasattr(test_dataset, 'get_class_names'):
            class_names_list = test_dataset.get_class_names()
            return {i: name for i, name in enumerate(class_names_list)}
        else:
            # Fallback: collect from dataset samples
            label_to_name = {}
            for i in range(min(len(test_dataset), 1000)):  # Sample to avoid loading all
                _, label, label_name = test_dataset[i]
                if isinstance(label, (int, np.integer)):
                    label = int(label)
                if label not in label_to_name:
                    label_to_name[label] = label_name
            return label_to_name

    def _save_detailed_predictions(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        class_names_mapping: dict,
        experiment_name: str,
        filename_prefix: str
    ) -> None:
        """
        Save per-sample predictions to CSV and print confused class pairs.
        
        Args:
            predictions: Predicted labels
            labels: Ground truth labels
            class_names_mapping: class_id -> class_name
            experiment_name: e.g. "Synthetic Only"
            filename_prefix: e.g. "detailed_predictions_synthetic_only"
        """
        csv_path = self.run_dir / f"{filename_prefix}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_idx', 'true_label', 'true_class_name',
                'pred_label', 'pred_class_name', 'correct'
            ])
            for i in range(len(labels)):
                true_id = int(labels[i])
                pred_id = int(predictions[i])
                true_name = class_names_mapping.get(true_id, f"class_{true_id}")
                pred_name = class_names_mapping.get(pred_id, f"class_{pred_id}")
                writer.writerow([
                    i, true_id, true_name, pred_id, pred_name,
                    'yes' if true_id == pred_id else 'no'
                ])
        print(f"\n[Detailed] Per-sample predictions saved to: {csv_path}")

        # Count confusions: (true_class, pred_class) -> count
        confusion_pairs = defaultdict(int)
        for true_id, pred_id in zip(labels, predictions):
            true_id, pred_id = int(true_id), int(pred_id)
            if true_id != pred_id:
                confusion_pairs[(true_id, pred_id)] += 1

        if not confusion_pairs:
            print(f"[Detailed] No misclassifications in {experiment_name}.")
            return

        # Sort by count descending and show top confused pairs
        sorted_pairs = sorted(
            confusion_pairs.items(),
            key=lambda x: -x[1]
        )
        top_n = min(20, len(sorted_pairs))
        print(f"\n[Detailed] Top {top_n} confused (true -> pred) in {experiment_name}:")
        for (true_id, pred_id), count in sorted_pairs[:top_n]:
            true_name = class_names_mapping.get(true_id, f"class_{true_id}")
            pred_name = class_names_mapping.get(pred_id, f"class_{pred_id}")
            print(f"  {true_name} -> {pred_name}: {count} 次")
    
    def run_classifier_experiments(self) -> dict:
        """Run classifier training experiments (Synthetic Only)."""
        print("\n" + "=" * 60)
        print("CLIP + Probe Experiments (Logistic Regression & MLP)")
        print("=" * 60)

        syn_features, syn_labels = self._features['syn_train']
        test_features, test_labels = self._features['real_test']

        # Get class names mapping
        class_names_mapping = self._get_class_names_mapping()

        max_iter = self.config['training']['max_iter']
        results = {}

        eval_class_indices = np.unique(syn_labels)

        # ----- Logistic Regression probe -----
        print("\n" + "-" * 40)
        print("Training LR on SYNTHETIC data only")
        print("-" * 40)

        lr_trainer = CLIPClassifierTrainer(
            max_iter=max_iter,
            random_state=self.seed,
            verbose=1,
            eval_class_indices=eval_class_indices,
        )
        lr_accuracy, lr_eval_results = lr_trainer.train_and_evaluate(
            syn_features, syn_labels, test_features, test_labels
        )
        results['synthetic_only_lr'] = lr_accuracy
        results['synthetic_only'] = lr_accuracy  # backward-compatible alias
        print(f"*** LR Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%) ***")

        lr_per_class_acc = self._calculate_per_class_accuracy(
            lr_eval_results['predictions'],
            lr_eval_results['labels'],
            class_names=class_names_mapping,
        )

        self.results.add_result(
            name="Synthetic Only (LR)",
            accuracy=lr_accuracy,
            num_train_samples=len(syn_labels),
            num_test_samples=len(test_labels),
            per_class_accuracy=lr_per_class_acc,
        )

        if self.detailed_results:
            self._save_detailed_predictions(
                lr_eval_results['predictions'],
                lr_eval_results['labels'],
                class_names_mapping,
                experiment_name="Synthetic Only - LR (trained model)",
                filename_prefix="detailed_predictions_synthetic_only_lr",
            )

        # ----- MLP probe (mirrors AttrSyn/clip_probe.py) -----
        print("\n" + "-" * 40)
        print("Training MLP on SYNTHETIC data only")
        print("-" * 40)

        mlp_trainer = CLIPMLPClassifierTrainer(
            eval_class_indices=eval_class_indices,
        )
        mlp_accuracy, mlp_eval_results = mlp_trainer.train_and_evaluate(
            syn_features, syn_labels, test_features, test_labels
        )
        results['synthetic_only_mlp'] = mlp_accuracy
        print(f"*** MLP Accuracy: {mlp_accuracy:.4f} ({mlp_accuracy*100:.2f}%) ***")

        mlp_per_class_acc = self._calculate_per_class_accuracy(
            mlp_eval_results['predictions'],
            mlp_eval_results['labels'],
            class_names=class_names_mapping,
        )

        self.results.add_result(
            name="Synthetic Only (MLP)",
            accuracy=mlp_accuracy,
            num_train_samples=len(syn_labels),
            num_test_samples=len(test_labels),
            per_class_accuracy=mlp_per_class_acc,
        )

        if self.detailed_results:
            self._save_detailed_predictions(
                mlp_eval_results['predictions'],
                mlp_eval_results['labels'],
                class_names_mapping,
                experiment_name="Synthetic Only - MLP (trained model)",
                filename_prefix="detailed_predictions_synthetic_only_mlp",
            )

        return results
    
    def run(self, skip_zero_shot: bool = False) -> None:
        """
        Run the complete evaluation pipeline.
        
        Args:
            skip_zero_shot: Whether to skip zero-shot evaluation
        """
        # Print run directory information
        print("\n" + "=" * 70)
        print(f"Run Directory: {self.run_dir}")
        print(f"  Features will be saved to: {self.features_dir}")
        print(f"  Results will be saved to: {self.run_dir}")
        print(f"  Shared test feature cache: {self.test_features_path}")
        print("=" * 70)
        
        # Load datasets
        self.load_datasets()

        # Set metadata after dataset loading so class information is populated.
        self.results.set_metadata(
            model=self.config['model']['vision_encoder'],
            dataset_name=self.dataset_config.get('name', 'dataset'),
            dataset_loader=self.dataset_config.get('loader', 'huggingface'),
            dataset_path=self.dataset_config.get('path'),
            syn_index=self.syn_index_path or self.synthetic_root,
            n_per_class=self.n_per_class,
            syn_samples=self.syn_samples or 'all',
            num_eval_classes=len(self.eval_original_labels),
            eval_original_labels=self.eval_original_labels,
            eval_class_names=self.active_class_names,
            seed=self.seed,
            test_feature_cache=str(self.test_features_path),
            run_dir=str(self.run_dir)
        )
        
        # Extract features
        self.extract_features()
        
        # Run zero-shot
        if not skip_zero_shot:
            self.run_zero_shot()
        
        # Run classifier experiments
        self.run_classifier_experiments()
        
        # Print and save results
        self.results.print_summary()
        self.results.save_results(filename_prefix="clip_eval")


def main():
    """Main entry point."""
    args = parse_args()

    # Print header
    print("=" * 70)
    print("Synthetic Data Evaluation Pipeline")
    print("=" * 70)
    print(f"Working directory: {os.getcwd()}")
    print(f"\nConfiguration:")
    if args.syn_index:
        print(f"  Synthetic index (JSON): {args.syn_index}")
    else:
        print(f"  Synthetic root (dir):   {args.synthetic_root}")
    print(f"  Samples per class: {args.n_per_class}")
    print(f"  Synthetic samples: {args.syn_samples or 'all'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Resize: {args.resize or 'original size'}")
    print(f"  Random seed: {args.seed}")
    print(f"  Detailed results (per-sample): {args.detailed_results}")

    # Load config
    config = load_config(args.config)

    # CLI override for backbone (accepts OpenAI-style names too)
    if args.clip_model:
        config.setdefault('model', {})['vision_encoder'] = args.clip_model
        print(f"  CLIP model (override): {args.clip_model}")

    # CLI override for open_clip pretrained tag
    if args.clip_pretrained:
        config.setdefault('model', {})['vision_encoder_pretrained'] = args.clip_pretrained
        print(f"  CLIP pretrained (open_clip): {args.clip_pretrained}")

    # Explicit backend selection (required, no auto-routing)
    config.setdefault('model', {})['vision_encoder_backend'] = args.clip_backend
    print(f"  CLIP backend: {args.clip_backend}")
    if args.clip_backend == 'open_clip' and not args.clip_pretrained:
        sys.exit(
            "Error: --clip_backend=open_clip requires --clip_pretrained (e.g. laion2b_s32b_b82k). "
            "Run `python -c \"import open_clip; print(open_clip.list_pretrained())\"` to see all options."
        )

    # Validate input paths
    if args.syn_index and not os.path.exists(args.syn_index):
        sys.exit(f"Error: Synthetic index file not found: {args.syn_index}")
    if args.synthetic_root and not os.path.isdir(args.synthetic_root):
        sys.exit(f"Error: Synthetic root directory not found: {args.synthetic_root}")

    # Create and run evaluator
    evaluator = SyntheticDataEvaluator(
        config=config,
        syn_index_path=args.syn_index,
        synthetic_root=args.synthetic_root,
        n_per_class=args.n_per_class,
        syn_samples=args.syn_samples,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
        features_dir=args.features_dir,
        seed=args.seed,
        detailed_results=args.detailed_results,
        resize=args.resize
    )

    evaluator.run(skip_zero_shot=args.no_zero_shot)


if __name__ == "__main__":
    main()

