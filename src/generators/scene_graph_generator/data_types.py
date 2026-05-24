"""
Data types for Scene Graph Generator.

This module defines dataclasses for representing scene graphs,
objects, relations, and generation results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


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
    """Represents a complete scene graph item with caption."""
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
class CoObjectItem:
    """Represents an item from the co-objects input file with three categories."""
    label: int
    label_name: str
    subject: str
    semantically_associated: List[str] = field(default_factory=list)
    compatible_non_typical: List[str] = field(default_factory=list)
    contextually_contrastive: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoObjectItem":
        """Create from dictionary."""
        return cls(
            label=data["label"],
            label_name=data["label_name"],
            subject=data["subject"],
            semantically_associated=data.get("semantically_associated", []),
            compatible_non_typical=data.get("compatible_non_typical", []),
            contextually_contrastive=data.get("contextually_contrastive", [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "label": self.label,
            "label_name": self.label_name,
            "subject": self.subject,
            "semantically_associated": self.semantically_associated,
            "compatible_non_typical": self.compatible_non_typical,
            "contextually_contrastive": self.contextually_contrastive
        }
    
    def get_all_objects(self) -> List[str]:
        """Get all objects from all categories as a flat list."""
        all_objects = []
        all_objects.extend(self.semantically_associated)
        all_objects.extend(self.compatible_non_typical)
        all_objects.extend(self.contextually_contrastive)
        return all_objects


@dataclass
class GenerationTask:
    """Represents a single generation task."""
    task_id: int
    subject: str
    sampled_objects: List[str]
    label: int
    label_name: str
    
    def to_prompt_info(self) -> str:
        """Generate information string for prompt."""
        objects_str = ", ".join(self.sampled_objects)
        return f"Subject: {self.subject}\nObjects: {objects_str}"


@dataclass
class SyntheticDataset:
    """Represents the complete synthetic dataset output."""
    dataset: str
    total: int
    results: List[SceneGraphItem] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset": self.dataset,
            "total": self.total,
            "results": [item.to_dict() for item in self.results]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyntheticDataset":
        """Create from dictionary."""
        return cls(
            dataset=data["dataset"],
            total=data["total"],
            results=[SceneGraphItem.from_dict(item) for item in data.get("results", [])]
        )

