"""
Data types for Image Generator.

This module defines dataclasses for representing image generation tasks,
results, and related data structures.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class ObjectEntity:
    """Represents an object entity in a scene graph."""
    id: int
    name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectEntity":
        """Create from dictionary."""
        return cls(id=data["id"], name=data["name"])


@dataclass
class Relation:
    """Represents a relation between subject and object."""
    subject_id: int
    object_id: int
    relation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "subject_id": self.subject_id,
            "object_id": self.object_id,
            "relation": self.relation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        """Create from dictionary."""
        return cls(
            subject_id=data["subject_id"],
            object_id=data["object_id"],
            relation=data["relation"]
        )


@dataclass
class SceneGraph:
    """Represents a scene graph with subject, objects, and relations."""
    id: int
    subject: ObjectEntity
    objects: List[ObjectEntity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "subject": self.subject.to_dict(),
            "objects": [obj.to_dict() for obj in self.objects],
            "relations": [rel.to_dict() for rel in self.relations]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneGraph":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            subject=ObjectEntity.from_dict(data["subject"]),
            objects=[ObjectEntity.from_dict(obj) for obj in data.get("objects", [])],
            relations=[Relation.from_dict(rel) for rel in data.get("relations", [])]
        )


@dataclass
class SceneGraphItem:
    """Represents a scene graph item from input JSON."""
    id: int
    label: int
    label_name: str
    scene_graph: SceneGraph
    caption: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "label": self.label,
            "label_name": self.label_name,
            "scene_graph": self.scene_graph.to_dict(),
            "caption": self.caption
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneGraphItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            label=data["label"],
            label_name=data["label_name"],
            scene_graph=SceneGraph.from_dict(data["scene_graph"]),
            caption=data["caption"]
        )


@dataclass
class ImageGenerationTask:
    """Represents a single image generation task."""
    task_id: int
    caption: str
    label: int
    label_name: str
    scene_graph: SceneGraph
    
    @classmethod
    def from_scene_graph_item(cls, item: SceneGraphItem) -> "ImageGenerationTask":
        """Create from SceneGraphItem."""
        return cls(
            task_id=item.id,
            caption=item.caption,
            label=item.label,
            label_name=item.label_name,
            scene_graph=item.scene_graph
        )


@dataclass
class ImageGenerationResult:
    """Represents the result of an image generation task."""
    id: int
    label: int
    label_name: str
    scene_graph: SceneGraph
    caption: str
    image_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "label": self.label,
            "label_name": self.label_name,
            "scene_graph": self.scene_graph.to_dict(),
            "caption": self.caption,
            "image_path": self.image_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageGenerationResult":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            label=data["label"],
            label_name=data["label_name"],
            scene_graph=SceneGraph.from_dict(data["scene_graph"]),
            caption=data["caption"],
            image_path=data.get("image_path")
        )
    
    @classmethod
    def from_task(
        cls,
        task: ImageGenerationTask,
        image_path: Optional[str] = None
    ) -> "ImageGenerationResult":
        """Create from ImageGenerationTask."""
        return cls(
            id=task.task_id,
            label=task.label,
            label_name=task.label_name,
            scene_graph=task.scene_graph,
            caption=task.caption,
            image_path=image_path
        )


@dataclass
class SyntheticImageDataset:
    """Represents the complete synthetic image dataset output."""
    dataset: str
    total: int
    image_dir: str
    results: List[ImageGenerationResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset": self.dataset,
            "total": self.total,
            "image_dir": self.image_dir,
            "results": [item.to_dict() for item in self.results]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyntheticImageDataset":
        """Create from dictionary."""
        return cls(
            dataset=data["dataset"],
            total=data["total"],
            image_dir=data.get("image_dir", ""),
            results=[ImageGenerationResult.from_dict(item) for item in data.get("results", [])]
        )


@dataclass
class GeneratorConfig:
    """Configuration for the image generator."""
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    device: str = "cuda"
    torch_dtype: str = "float16"  # "float16" or "float32"
    seed: int = 42
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    image_width: int = 1024
    image_height: int = 1024
    image_format: str = "jpg"  # "jpg" or "png"
    filename_prefix: str = "syn_image"
    use_safetensors: bool = True
    enable_xformers: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_format": self.image_format,
            "filename_prefix": self.filename_prefix,
            "use_safetensors": self.use_safetensors,
            "enable_xformers": self.enable_xformers
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratorConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

