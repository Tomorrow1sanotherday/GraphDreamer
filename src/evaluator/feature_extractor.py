"""
CLIP Feature Extractor for image representation learning.

Provides utilities to extract visual and text features using CLIP models
for downstream classification tasks.

Supports three backends, selected by (model_name, pretrained):
  - open_clip: when `pretrained` is provided (LAION / DataComp / etc. weights)
      e.g. model_name="ViT-L-14", pretrained="laion2b_s32b_b82k"
  - OpenAI clip: when model_name matches the OpenAI clip-package names
      e.g. "RN50", "RN101", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"
  - HuggingFace transformers: any other identifier
      e.g. "openai/clip-vit-large-patch14"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# OpenAI CLIP model names (loaded via `import clip`); mirror clip.available_models()
OPENAI_CLIP_NAMES = {
    "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64",
    "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px",
}


def is_openai_clip_name(model_name: str) -> bool:
    """Return True if model_name is one of the OpenAI clip-package names."""
    return model_name in OPENAI_CLIP_NAMES


def pil_collate_fn(batch: List[Tuple]) -> Tuple[List[Image.Image], torch.Tensor, List[str]]:
    """
    Custom collate function that keeps PIL images as a list.

    Args:
        batch: List of (image, label, label_name) tuples

    Returns:
        Tuple of (images_list, labels_tensor, label_names_list)
    """
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    label_names = [item[2] for item in batch]
    return images, labels, label_names


class CLIPFeatureExtractor:
    """
    CLIP-based feature extractor for images and text.

    Supports:
    - Batch extraction of image features
    - Dataset-level feature extraction with caching
    - Text feature extraction for zero-shot classification
    """

    SUPPORTED_BACKENDS = ("openai_clip", "open_clip", "huggingface")

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None,
        pretrained: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        """
        Initialize CLIP feature extractor.

        Args:
            model_name: Either an OpenAI clip name (e.g. "ViT-L/14", "RN50"),
                an open_clip architecture (e.g. "ViT-L-14"), or a HuggingFace
                identifier (e.g. "openai/clip-vit-large-patch14").
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            pretrained: open_clip pretrained tag (e.g. "laion2b_s32b_b82k").
                Required when backend='open_clip'; ignored by the other two backends.
            backend: One of "openai_clip" / "open_clip" / "huggingface". When set,
                forces that backend and skips name-based auto-routing. When None,
                falls back to legacy auto-routing: pretrained non-empty → open_clip,
                else OPENAI_CLIP_NAMES match → openai_clip, else huggingface.
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if backend is not None:
            if backend not in self.SUPPORTED_BACKENDS:
                raise ValueError(
                    f"backend must be one of {self.SUPPORTED_BACKENDS}, got {backend!r}"
                )
            if backend == "open_clip" and not pretrained:
                raise ValueError(
                    "backend='open_clip' requires `pretrained` (e.g. 'laion2b_s32b_b82k'). "
                    "See `python -c \"import open_clip; print(open_clip.list_pretrained())\"`."
                )
            self.backend = backend
        elif pretrained:
            self.backend = "open_clip"
        elif is_openai_clip_name(model_name):
            self.backend = "openai_clip"
        else:
            self.backend = "huggingface"

        backend_label = f"{self.backend}"
        if self.backend == "open_clip":
            backend_label += f" (pretrained={pretrained})"
        print(f"Loading CLIP model: {model_name} (backend: {backend_label})")
        print(f"Using device: {self.device}")

        if self.backend == "open_clip":
            import open_clip  # lazy import; installed via `pip install open_clip_torch`
            self._open_clip_module = open_clip
            self.model, _, self._open_clip_preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            self.model.eval()
            self._open_clip_tokenizer = open_clip.get_tokenizer(model_name)
            self.processor = None
            with torch.no_grad():
                dummy = self._open_clip_tokenizer(["a"]).to(self.device)
                self._feature_dim = self.model.encode_text(dummy).shape[-1]
        elif self.backend == "openai_clip":
            import clip  # lazy import; package shipped with the copulasyn env
            self._clip_module = clip
            self.model, self._openai_preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
            # Match HF API surface so callers can still reference these attributes.
            self.processor = None
            with torch.no_grad():
                dummy = clip.tokenize(["a"]).to(self.device)
                self._feature_dim = self.model.encode_text(dummy).shape[-1]
        else:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
            self._feature_dim = self.model.config.projection_dim

        print(f"Feature dimension: {self._feature_dim}")

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension of the model."""
        return self._feature_dim

    @torch.no_grad()
    def extract_image_features(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extract features from a batch of PIL images.

        Args:
            images: List of PIL Images

        Returns:
            numpy array of shape (N, feature_dim)
        """
        if self.backend == "open_clip":
            batch = torch.stack([self._open_clip_preprocess(img) for img in images]).to(self.device)
            features = self.model.encode_image(batch)
            return features.float().cpu().numpy()

        if self.backend == "openai_clip":
            # Apply CLIP's official preprocess to each image, then stack.
            batch = torch.stack([self._openai_preprocess(img) for img in images]).to(self.device)
            features = self.model.encode_image(batch)
            return features.float().cpu().numpy()

        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        features = self.model.get_image_features(**inputs)
        return features.cpu().numpy()

    @torch.no_grad()
    def extract_text_features(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Extract features from a list of text prompts.

        Args:
            texts: List of text strings
            normalize: Whether to L2-normalize features

        Returns:
            Tensor of shape (N, feature_dim) on the extractor device.
        """
        if self.backend == "open_clip":
            tokens = self._open_clip_tokenizer(texts).to(self.device)
            features = self.model.encode_text(tokens).float()
        elif self.backend == "openai_clip":
            tokens = self._clip_module.tokenize(texts).to(self.device)
            features = self.model.encode_text(tokens).float()
        else:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_text_features(**inputs)

        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)

        return features

    @torch.no_grad()
    def encode_images_normalized(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Encode a batch of PIL images and return L2-normalized features as a
        torch.Tensor on the extractor device. Backend-agnostic helper used by
        the zero-shot evaluator (which needs to multiply by text features).
        """
        if self.backend == "open_clip":
            batch = torch.stack([self._open_clip_preprocess(img) for img in images]).to(self.device)
            features = self.model.encode_image(batch).float()
        elif self.backend == "openai_clip":
            batch = torch.stack([self._openai_preprocess(img) for img in images]).to(self.device)
            features = self.model.encode_image(batch).float()
        else:
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_image_features(**inputs)

        return features / features.norm(dim=-1, keepdim=True)

    def extract_dataset_features(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        save_path: Optional[str] = None,
        num_workers: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract features from an entire dataset.

        Args:
            dataset: Dataset returning (image, label, label_name) tuples
            batch_size: Batch size for extraction
            save_path: Optional path to save extracted features
            num_workers: Number of data loading workers

        Returns:
            Tuple of (features, labels, label_names)
        """
        # Check for cached features
        if save_path and os.path.exists(save_path):
            print(f"Loading cached features from {save_path}")
            data = np.load(save_path, allow_pickle=True)
            return data['features'], data['labels'], list(data['label_names'])

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=pil_collate_fn
        )

        all_features = []
        all_labels = []
        all_label_names = []

        print(f"Extracting features from {len(dataset)} samples...")

        for batch_images, batch_labels, batch_label_names in tqdm(dataloader):
            features = self.extract_image_features(batch_images)

            all_features.append(features)
            all_labels.extend(batch_labels.numpy())
            all_label_names.extend(batch_label_names)

        all_features = np.vstack(all_features)
        all_labels = np.array(all_labels)

        print(f"Extracted features shape: {all_features.shape}")

        # Save features if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(
                save_path,
                features=all_features,
                labels=all_labels,
                label_names=all_label_names
            )
            print(f"Features saved to {save_path}")

        return all_features, all_labels, all_label_names
