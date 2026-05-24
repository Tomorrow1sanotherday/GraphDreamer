"""
CLIP-based evaluators for image classification.

Provides:
- CLIPZeroShotEvaluator: Zero-shot classification using text-image similarity
- CLIPClassifierTrainer: Logistic regression on CLIP features
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .feature_extractor import CLIPFeatureExtractor, pil_collate_fn


class CLIPZeroShotEvaluator:
    """
    Zero-shot image classifier using CLIP text-image similarity.

    Uses text prompts of the form "a {domain} of a {class_name}" to classify
    images without any training data (e.g. "a photo of a {c}",
    "a painting of a {c}").
    """

    def __init__(
        self,
        extractor: CLIPFeatureExtractor,
        domain: str,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize zero-shot evaluator.

        Args:
            extractor: CLIPFeatureExtractor instance
            domain: Image-domain word inserted as {domain} (e.g. "photo",
                "painting", "sketch"). Used to build the default template
                "a {domain} of a {class}".
            prompt_template: Optional full template override with `{}` as the
                class-name placeholder. If None, built from `domain`.
        """
        self.extractor = extractor
        self.domain = domain
        if prompt_template is None:
            prompt_template = "a {}"
        self.prompt_template = prompt_template
        self._text_features = None
        self._class_names = None
    
    def _prepare_text_features(self, class_names: List[str]) -> None:
        """
        Pre-compute text features for all classes.
        
        Args:
            class_names: List of class names
        """
        if self._class_names == class_names:
            return  # Already computed
        
        self._class_names = class_names
        
        # Create text prompts
        prompts = [
            self.prompt_template.format(name)
            for name in class_names
        ]
        
        print(f"Encoding {len(prompts)} class prompts...")
        self._text_features = self.extractor.extract_text_features(prompts, normalize=True)
        print(f"Text features shape: {self._text_features.shape}")
    
    def evaluate(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Tuple[float, dict]:
        """
        Perform zero-shot evaluation on a dataset.
        
        Args:
            dataset: Dataset returning (image, label, label_name) tuples
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
        
        Returns:
            Tuple of (accuracy, detailed_results)
        """
        # Collect class names from dataset
        class_names = self._collect_class_names(dataset)
        self._prepare_text_features(class_names)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=pil_collate_fn
        )
        
        all_predictions = []
        all_labels = []

        print("Evaluating zero-shot classification...")

        with torch.no_grad():
            for batch_images, batch_labels, _ in tqdm(dataloader):
                # Backend-agnostic: extractor handles HF vs OpenAI-clip preprocessing.
                image_features = self.extractor.encode_images_normalized(batch_images)

                # Compute similarity and predict
                similarity = 100.0 * image_features @ self._text_features.T
                predictions = similarity.argmax(dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'num_samples': len(all_labels),
            'num_classes': len(class_names),
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        return accuracy, results
    
    def _collect_class_names(self, dataset: Dataset) -> List[str]:
        """Collect all class names from dataset."""
        if hasattr(dataset, 'get_class_names'):
            return dataset.get_class_names()
        
        # Fallback: iterate through dataset
        print("Collecting class names from dataset...")
        label_to_name = {}
        for i in range(len(dataset)):
            _, label, label_name = dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label not in label_to_name:
                label_to_name[label] = label_name
            if len(label_to_name) >= 200:  # Early stop for CUB-200
                break
        
        max_label = max(label_to_name.keys())
        return [label_to_name.get(i, f"class_{i}") for i in range(max_label + 1)]


class CLIPClassifierTrainer:
    """
    Logistic Regression classifier trained on CLIP features.
    
    Provides a simple linear probe baseline for evaluating
    feature quality from synthetic data.
    """
    
    def __init__(
        self,
        max_iter: int = 1000,
        random_state: int = 42,
        verbose: int = 1,
        eval_class_indices: np.ndarray = None
    ):
        """
        Initialize classifier trainer.

        Args:
            max_iter: Maximum iterations for logistic regression
            random_state: Random seed for reproducibility
            verbose: Verbosity level (0=silent, 1=progress)
            eval_class_indices: If set, at predict time only output among these classes
                               (e.g. 0-199; rubbish/200 is masked out).
        """
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.eval_class_indices = eval_class_indices
        self.classifier = None
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> 'CLIPClassifierTrainer':
        """
        Train logistic regression classifier.
        
        Args:
            features: Feature array of shape (N, D)
            labels: Label array of shape (N,)
        
        Returns:
            self for method chaining
        """
        print(f"Training classifier on {len(labels)} samples...")
        
        self.classifier = LogisticRegression(
            C=0.316,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=self.verbose
        )
        self.classifier.fit(features, labels)
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        If eval_class_indices is set, only classes in that set can be predicted
        (e.g. mask rubbish class: take argmax among 0-199).
        
        Args:
            features: Feature array of shape (N, D)
        
        Returns:
            Predicted labels of shape (N,)
        """
        if self.classifier is None:
            raise RuntimeError("Classifier not trained. Call train() first.")
        if self.eval_class_indices is None:
            return self.classifier.predict(features)
        proba = self.classifier.predict_proba(features)
        # Mask out classes not in eval_class_indices: set their logits to -inf
        mask = np.isin(self.classifier.classes_, self.eval_class_indices)
        proba_masked = np.full_like(proba, -np.inf)
        proba_masked[:, mask] = proba[:, mask]
        pred_indices = np.argmax(proba_masked, axis=1)
        return self.classifier.classes_[pred_indices]
    
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Tuple[float, dict]:
        """
        Evaluate classifier on test data.
        
        Args:
            features: Feature array of shape (N, D)
            labels: Ground truth labels of shape (N,)
        
        Returns:
            Tuple of (accuracy, detailed_results)
        """
        predictions = self.predict(features)
        accuracy = accuracy_score(labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'num_samples': len(labels),
            'predictions': predictions,
            'labels': labels
        }
        
        return accuracy, results
    
    def train_and_evaluate(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray
    ) -> Tuple[float, dict]:
        """
        Train classifier and evaluate on test set.
        
        Args:
            train_features: Training features
            train_labels: Training labels
            test_features: Test features
            test_labels: Test labels
        
        Returns:
            Tuple of (accuracy, detailed_results)
        """
        self.train(train_features, train_labels)
        return self.evaluate(test_features, test_labels)


class CLIPMLPClassifierTrainer:
    """
    MLP classifier trained on CLIP features.

    Mirrors the MLP probe in AttrSyn/clip_probe.py:
        MLPClassifier(hidden_layer_sizes=(256,), activation='relu',
                      max_iter=1000, verbose=0)
    """

    def __init__(
        self,
        eval_class_indices: np.ndarray = None,
    ):
        """
        Args:
            eval_class_indices: If set, at predict time only output among these classes.
        """
        self.eval_class_indices = eval_class_indices
        self.classifier = None

    def train(self, features: np.ndarray, labels: np.ndarray) -> 'CLIPMLPClassifierTrainer':
        print(f"Training MLP classifier on {len(labels)} samples...")

        self.classifier = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation='relu',
            max_iter=1000,
            verbose=0,
        )
        self.classifier.fit(features, labels)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Classifier not trained. Call train() first.")
        if self.eval_class_indices is None:
            return self.classifier.predict(features)
        proba = self.classifier.predict_proba(features)
        mask = np.isin(self.classifier.classes_, self.eval_class_indices)
        proba_masked = np.full_like(proba, -np.inf)
        proba_masked[:, mask] = proba[:, mask]
        pred_indices = np.argmax(proba_masked, axis=1)
        return self.classifier.classes_[pred_indices]

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Tuple[float, dict]:
        predictions = self.predict(features)
        accuracy = accuracy_score(labels, predictions)

        results = {
            'accuracy': accuracy,
            'num_samples': len(labels),
            'predictions': predictions,
            'labels': labels,
        }

        return accuracy, results

    def train_and_evaluate(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
    ) -> Tuple[float, dict]:
        self.train(train_features, train_labels)
        return self.evaluate(test_features, test_labels)


def sample_features_per_class(
    features: np.ndarray,
    labels: np.ndarray,
    n_per_class: int = 3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample n feature vectors per class for few-shot evaluation.
    
    Args:
        features: Feature array of shape (N, D)
        labels: Label array of shape (N,)
        n_per_class: Number of samples per class
        random_state: Random seed
    
    Returns:
        Tuple of (sampled_features, sampled_labels)
    """
    np.random.seed(random_state)
    
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    sampled_indices = []
    for label in sorted(class_indices.keys()):
        indices = class_indices[label]
        if len(indices) >= n_per_class:
            sampled = np.random.choice(indices, n_per_class, replace=False)
        else:
            sampled = np.array(indices)
        sampled_indices.extend(sampled)
    
    sampled_indices = np.array(sampled_indices)
    return features[sampled_indices], labels[sampled_indices]


def combine_features(
    feature_sets: List[np.ndarray],
    label_sets: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine multiple feature and label arrays.
    
    Args:
        feature_sets: List of feature arrays
        label_sets: List of label arrays
    
    Returns:
        Tuple of (combined_features, combined_labels)
    """
    combined_features = np.vstack(feature_sets)
    combined_labels = np.concatenate(label_sets)
    return combined_features, combined_labels

