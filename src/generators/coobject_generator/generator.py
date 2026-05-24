"""
Co-occurrence Object Generator Module

This module provides functionality to generate background objects that can
co-occur with given subjects (labels) in images, using LLM (DeepSeek).

Background objects are generated in three categories:
1. Semantically associated: Background objects commonly co-occurring with the subject
2. Compatible but non-typical: Background objects that could appear alongside the subject but are not typically associated
3. Contextually contrastive: Background objects semantically unrelated to the subject but physically feasible to co-exist
"""

import json
import re
import logging
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)


class CategoryType(Enum):
    """Enum for the three categories of co-occurring objects."""
    SEMANTICALLY_ASSOCIATED = "semantically_associated"
    COMPATIBLE_NON_TYPICAL = "compatible_non_typical"
    CONTEXTUALLY_CONTRASTIVE = "contextually_contrastive"


@dataclass
class LabelItem:
    """Represents a label item from the input file."""
    label: int
    label_name: str
    
    @property
    def display_name(self) -> str:
        """Get the display name with underscores replaced by spaces."""
        return self.label_name.replace("_", " ")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabelItem":
        """Create a LabelItem from a dictionary."""
        return cls(
            label=data["label"],
            label_name=data["label_name"]
        )


@dataclass
class CooccurrenceResult:
    """Represents the co-occurrence result for a single subject with three categories."""
    label: int
    label_name: str
    subject: str
    semantically_associated: List[str] = field(default_factory=list)
    compatible_non_typical: List[str] = field(default_factory=list)
    contextually_contrastive: List[str] = field(default_factory=list)
    
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


class ObjectCooccurrenceGenerator:
    """
    Generator for co-occurring objects using LLM.
    
    This class handles:
    - Loading labels from input JSON
    - Generating prompts for LLM (separate call for each category)
    - Parsing and validating LLM responses
    - Ensuring no duplicate objects
    - Saving results to output JSON
    """
    
    # Blacklist of invisible/abstract objects that cannot be rendered by text-to-image models
    INVISIBLE_OBJECTS_BLACKLIST = {
        # Natural phenomena (invisible)
        "wind", "breeze", "gust", "air", "atmosphere", "temperature", "humidity", 
        "weather", "climate", "pressure", "moisture", "vapor", "steam",
        "rain", "storm", "fog", "mist", "hail",
        # Note: "snow" as falling is invisible, but "snow" as ground/visible surface is allowed
        # Abstract concepts
        "time", "space", "energy", "sound", "silence", "darkness",
        # Emotions/states
        "happiness", "sadness", "fear", "peace", "chaos", "calm", "tension",
        # Other abstract concepts
        "nothing", "void", "emptiness", "infinity", "eternity"
    }
    
    # Allowed compound words that contain blacklisted terms but refer to visible objects
    ALLOWED_COMPOUND_WORDS = {
        "rainbow", "raincoat", "rainwater", "windmill", "windshield", "airplane", 
        "airport", "aircraft", "storm drain", "fog light", "lamp", "light fixture",
        "lightbulb", "streetlight", "traffic light", "headlight", "spotlight"
    }
    
    # Animals that should always be filtered from background settings
    COMMON_ANIMALS = {
        "dog", "cat", "horse", "cow", "pig", "sheep", "goat", "deer", "bear",
        "wolf", "fox", "rabbit", "squirrel", "mouse", "rat", "hamster", "guinea pig",
        "lion", "tiger", "elephant", "zebra", "giraffe", "monkey", "ape", "chimpanzee",
        "gorilla", "orangutan", "kangaroo", "koala", "panda", "seal",
        "whale", "dolphin", "shark", "fish", "snake", "lizard", "turtle", "frog",
        "toad", "crocodile", "alligator",
        "bird", "birds", "eagle", "hawk", "owl", "crow", "raven", "sparrow",
        "robin", "pigeon", "dove", "duck", "goose", "swan", "chicken", "rooster",
        "turkey", "parrot", "canary", "finch", "cardinal", "blue jay", "woodpecker",
        "hummingbird", "seagull", "gull", "pelican", "flamingo", "penguin", "ostrich",
        "emu", "crane", "heron", "stork", "ibis", "egret", "albatross", "petrel",
        "tern", "gannet", "cormorant", "puffin", "guillemot", "grebe", "loon",
    }

    # Per-superclass generic terms that should be filtered as same-class members.
    # These catch generic words the LLM might produce that aren't in labels.json.
    SUPERCLASS_TERMS = {
        "bird": {
            "bird", "birds", "avian", "fowl", "poultry", "waterfowl", "songbird",
            "raptor", "seabird", "shorebird", "wading bird", "nest", "egg", "feather",
        },
        "aircraft": {
            "aircraft", "airplane", "aeroplane", "plane", "planes", "jet", "jets",
            "airliner", "jetliner", "biplane", "monoplane", "helicopter", "chopper",
            "glider", "bomber", "fighter jet", "propeller plane", "turboprop",
            "seaplane", "floatplane", "warplane",
        },
        "flower": {
            "flower", "flowers", "bloom", "blooms", "blossom", "blossoms",
            "petal", "petals", "bouquet", "floral", "wildflower", "wildflowers",
            "flowering plant", "seedling", "sprout",
        },
        "car": {
            "car", "cars", "automobile", "automobiles", "vehicle", "vehicles",
            "sedan", "coupe", "suv", "hatchback", "convertible", "wagon",
            "minivan", "pickup", "truck", "sports car", "supercar", "roadster",
            "limousine", "taxi", "cab", "bus", "van", "motorcycle", "scooter",
        },
    }
    
    # Base rules shared across all categories
    BASE_RULES = """Rules:
1. Output BACKGROUND SETTINGS only: habitat/landform/terrain/vegetation/water bodies/large static structures (i.e., the place/setting).
2. Concrete nouns only. No abstract/invisible concepts.
3. 1-2 words per item.
4. Do NOT output small standalone props (portable objects). If it feels like an object you can pick up, exclude it.
5. No people. No animals. No members of the same Superclass.
5. No body parts or biological remains (egg, feather, nest, bone).
6. No duplicates.

Return ONLY a JSON array of strings."""

    # Category-specific system prompts
    SYSTEM_PROMPTS = {
        CategoryType.SEMANTICALLY_ASSOCIATED: f"""Generate BACKGROUND SETTINGS strongly associated with the subject (typical habitat/setting). Output places/landforms/biomes, not props. {{base_rules}}""",

        CategoryType.COMPATIBLE_NON_TYPICAL: f"""Generate realistic BACKGROUND SETTINGS that could appear with the subject but are not typical for it (still plausible). Output places/landforms/biomes, not props. {{base_rules}}""",

        CategoryType.CONTEXTUALLY_CONTRASTIVE: f"""Generate BACKGROUND SETTINGS from a different setting than the subject (contrastive but still realistic). Output places/landforms/biomes, not props. {{base_rules}}"""
    }

    # All category types for iteration
    ALL_CATEGORIES = [
        CategoryType.SEMANTICALLY_ASSOCIATED,
        CategoryType.COMPATIBLE_NON_TYPICAL,
        CategoryType.CONTEXTUALLY_CONTRASTIVE
    ]

    def __init__(
        self,
        n_objects_per_category: int = 20,
        max_retries: int = 10,
        all_label_names: Optional[List[str]] = None,
    ):
        """
        Initialize the generator.

        Args:
            n_objects_per_category: Number of co-occurring objects to generate per category per subject
            max_retries: Maximum number of retries for validation failures
            all_label_names: All class names in the dataset — used to prevent
                the LLM from outputting members of the same class as co-objects.
        """
        self.n_objects_per_category = n_objects_per_category
        self.max_retries = max_retries
        self._dataset_class_words = self._build_class_word_set(all_label_names or [])

        logger.info(f"Initialized ObjectCooccurrenceGenerator with n_objects_per_category={n_objects_per_category}")
        if self._dataset_class_words:
            logger.info(f"Loaded {len(self._dataset_class_words)} dataset class words for same-class filtering")

    @staticmethod
    def _build_class_word_set(label_names: List[str]) -> set:
        """Build a lowercase set of individual words from all label names."""
        words: set = set()
        for name in label_names:
            normed = name.replace("_", " ").lower().strip()
            words.add(normed)
            for w in normed.split():
                if len(w) >= 3:
                    words.add(w)
        return words
    
    def load_labels(self, input_path: str) -> List[LabelItem]:
        """
        Load labels from a JSON file.
        
        Args:
            input_path: Path to the input JSON file
            
        Returns:
            List of LabelItem objects
        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        labels = [LabelItem.from_dict(item) for item in data]
        logger.info(f"Loaded {len(labels)} labels from {input_path}")
        
        return labels
    
    def build_prompt(
        self, 
        subject: str, 
        superclass: str = "Bird",
        category: CategoryType = None,
        exclude_objects: List[str] = None
    ) -> str:
        """
        Build the user prompt for LLM for a specific category.
        
        Args:
            subject: The subject name (with spaces, not underscores)
            superclass: The broad category of the subject (e.g., Bird, Vehicle)
            category: The category type to generate objects for
            exclude_objects: List of objects to exclude (from previous categories)
            
        Returns:
            The formatted user prompt
        """
        # Ask for extra objects as buffer (some may be filtered out)
        request_count = self.n_objects_per_category + 5

        category_line = f'\nCategory: "{category.value}"' if category is not None else ""
        
        prompt = f'''Subject: "{subject}"
Superclass: "{superclass}"{category_line}

Generate exactly {request_count} BACKGROUND SETTINGS for photos (the place/setting).
- Output places/landforms/biomes (terrain, vegetation, water, large structures), not standalone props.
- Concrete nouns only. 1-2 words each.
- No people. No animals. No members of the same Superclass.

Return ONLY a JSON array of strings.'''

        # Add exclusion list if provided
        if exclude_objects and len(exclude_objects) > 0:
            exclude_str = ", ".join(f'"{obj}"' for obj in exclude_objects)
            prompt += f'''

IMPORTANT: Do NOT include any of these objects (already used in other categories): [{exclude_str}]'''
        
        return prompt
    
    def create_validator(
        self, 
        subject: str,
        exclude_objects: List[str] = None,
        superclass: str = "Bird"
    ) -> Callable[[str], Optional[List[str]]]:
        """
        Create a validation function for LLM responses (single category).
        
        Args:
            subject: The subject name for validation
            exclude_objects: List of objects to exclude (from previous categories)
            superclass: The broad category of the subject (e.g., "Bird", "Vehicle")
            
        Returns:
            A validation function that returns parsed objects list or None if invalid
        """
        exclude_set = set(exclude_objects) if exclude_objects else set()

        # Build the combined filter set: common animals + superclass terms + dataset class names
        superclass_key = superclass.lower()
        same_class_filter = (
            self.COMMON_ANIMALS
            | self.SUPERCLASS_TERMS.get(superclass_key, set())
            | self._dataset_class_words
        )
        
        def validate(response: str) -> Optional[List[str]]:
            """
            Validate and parse the LLM response for a single category.
            
            Args:
                response: The raw LLM response string
                
            Returns:
                List of valid objects or None if validation fails
            """
            try:
                # Try to extract JSON array from response
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if not json_match:
                    logger.warning(f"No JSON array found in response: {response[:100]}...")
                    return None
                
                json_str = json_match.group()
                objects = json.loads(json_str)
                
                # Validate it's a list
                if not isinstance(objects, list):
                    logger.warning("Response is not a list")
                    return None
                
                # Validate all items are strings
                if not all(isinstance(obj, str) for obj in objects):
                    logger.warning("Not all items are strings")
                    return None
                
                subject_lower = subject.lower()
                subject_words = set(subject_lower.split())
                
                # Clean objects: remove underscores, strip whitespace
                cleaned_objects = []
                for obj in objects:
                    cleaned = obj.replace("_", " ").strip().lower()
                    if cleaned:
                        cleaned_objects.append(cleaned)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_objects = []
                for obj in cleaned_objects:
                    if obj not in seen:
                        seen.add(obj)
                        unique_objects.append(obj)
                
                # Filter out subject-related words, excluded objects, and invisible objects
                filtered_objects = []
                for obj in unique_objects:
                    # Skip if object contains the full subject name
                    if subject_lower in obj:
                        continue
                    # Skip if object is exactly one of the subject's words
                    if obj in subject_words:
                        continue
                    # Skip if object was already used in a previous category
                    if obj in exclude_set:
                        continue
                    # Skip if object is in the allowed compound words (these are visible objects)
                    if obj in self.ALLOWED_COMPOUND_WORDS:
                        filtered_objects.append(obj)
                        continue
                    
                    # Skip if object is in the invisible objects blacklist
                    if obj in self.INVISIBLE_OBJECTS_BLACKLIST:
                        logger.debug(f"Filtered out invisible object: {obj}")
                        continue
                    
                    # Skip if object is a single word that matches blacklist items
                    obj_words = obj.split()
                    if len(obj_words) == 1 and obj_words[0] in self.INVISIBLE_OBJECTS_BLACKLIST:
                        logger.debug(f"Filtered out invisible object: {obj}")
                        continue
                    
                    # Skip common invisible natural phenomena patterns (but allow compound words)
                    invisible_patterns = ["wind", "rain", "storm", "fog", "mist", "breeze", "gust", "air"]
                    if any(pattern in obj for pattern in invisible_patterns):
                        # Check if it's an allowed compound word
                        if obj not in self.ALLOWED_COMPOUND_WORDS:
                            logger.debug(f"Filtered out invisible natural phenomenon: {obj}")
                            continue
                    
                    # Skip if object matches a same-class term (dataset labels, superclass terms, animals)
                    if obj in same_class_filter:
                        logger.debug(f"Filtered out same-class term: {obj} (superclass: {superclass})")
                        continue

                    obj_words = obj.split()
                    if any(word in same_class_filter for word in obj_words):
                        logger.debug(f"Filtered out same-class term (word match): {obj} (superclass: {superclass})")
                        continue
                    
                    filtered_objects.append(obj)
                
                # Check we have enough objects after filtering
                if len(filtered_objects) < self.n_objects_per_category:
                    logger.warning(f"Not enough objects for '{subject}' after filtering: {len(filtered_objects)} < {self.n_objects_per_category}")
                    return None
                
                # Take only the required number
                return filtered_objects[:self.n_objects_per_category]
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
                return None
            except Exception as e:
                logger.warning(f"Validation error: {e}")
                return None
        
        return validate
    
    def prepare_prompts(self, labels: List[LabelItem], superclass: str = "Bird") -> List[tuple]:
        """
        Prepare prompts with indices for batch processing.
        
        Args:
            labels: List of LabelItem objects
            superclass: The broad category for all subjects (e.g., "Bird" for CUB-200)
            
        Returns:
            List of tuples (index, prompt)
        """
        prompts = []
        for i, label in enumerate(labels):
            subject = label.display_name
            prompt = self.build_prompt(subject, superclass)
            prompts.append((i, prompt))
        
        return prompts
    
    def iter_with_progress(
        self, 
        items: List[Any], 
        desc: str = "Processing",
        show_progress: bool = True
    ):
        """
        Iterate over items with optional progress bar.
        
        Args:
            items: List of items to iterate over
            desc: Description for progress bar
            show_progress: Whether to show progress bar
            
        Yields:
            Items from the list
        """
        if show_progress and HAS_TQDM:
            yield from tqdm(items, desc=desc, unit="item")
        else:
            total = len(items)
            for i, item in enumerate(items):
                if show_progress:
                    print(f"\r{desc}: {i+1}/{total} ({100*(i+1)/total:.1f}%)", end="", flush=True)
                yield item
            if show_progress and not HAS_TQDM:
                print()  # New line after progress
    
    def save_results(
        self,
        results: List[CooccurrenceResult],
        output_path: str
    ) -> None:
        """
        Save results to a JSON file.
        
        Args:
            results: List of CooccurrenceResult objects
            output_path: Path to the output JSON file
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data = [result.to_dict() for result in results]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Saved {len(results)} results to {output_path}")
    
    def get_system_prompt(self, category: CategoryType) -> str:
        """
        Get the system prompt for LLM for a specific category.
        
        Args:
            category: The category type to get the prompt for
            
        Returns:
            The formatted system prompt
        """
        template = self.SYSTEM_PROMPTS[category]
        return template.replace("{base_rules}", self.BASE_RULES)

