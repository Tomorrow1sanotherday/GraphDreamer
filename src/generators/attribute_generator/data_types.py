"""
Data types for attribute generation module.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class CoObjectItem:
    """Represents an item from the co-objects input file."""
    label: int
    label_name: str
    subject: str
    semantically_associated: List[str] = field(default_factory=list)
    compatible_non_typical: List[str] = field(default_factory=list)
    contextually_contrastive: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoObjectItem":
        """Create a CoObjectItem from a dictionary."""
        return cls(
            label=data["label"],
            label_name=data["label_name"],
            subject=data.get("subject", data["label_name"].replace("_", " ")),
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


@dataclass
class AttributeResult:
    """Represents the attribute result for a single subject."""
    label: int
    label_name: str
    subject: str
    semantically_associated: List[str] = field(default_factory=list)
    compatible_non_typical: List[str] = field(default_factory=list)
    contextually_contrastive: List[str] = field(default_factory=list)
    attributes: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "label": self.label,
            "label_name": self.label_name,
            "subject": self.subject,
            "semantically_associated": self.semantically_associated,
            "compatible_non_typical": self.compatible_non_typical,
            "contextually_contrastive": self.contextually_contrastive,
            "attributes": self.attributes
        }

