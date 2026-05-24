"""
Scene Graph Generator Module

This module provides functionality to generate scene graphs with relations
and captions using LLM for given subjects and their co-occurring objects.
"""

import json
import re
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple

from .data_types import (
    CoObjectItem,
    GenerationTask,
    SceneGraphItem,
    SceneGraph,
    ObjectEntity,
    Relation,
    SyntheticDataset
)

logger = logging.getLogger(__name__)


class SceneGraphGenerator:
    """
    Generator for scene graphs with relations and captions using LLM.
    
    This class handles:
    - Loading co-objects from input JSON
    - Sampling objects for each subject
    - Generating prompts for LLM
    - Parsing and validating LLM responses
    - Saving results with streaming output
    """

    ALLOWED_CATEGORIES = (
        "semantically_associated",
        "compatible_non_typical",
        "contextually_contrastive"
    )
    
    SYSTEM_PROMPT = """You are a scene graph and caption generator for image datasets.

Given a subject and a list of objects, generate:
1. Natural spatial/semantic relations between the subject and each object
2. A fluent, natural English caption describing the scene

Rules:
1. The subject always has id=0
2. Objects are numbered starting from id=1
3. Generate exactly ONE relation for each (subject, object) pair
4. Relations should be natural and visually plausible

5. CRITICAL - PURE RELATION FORMAT: The "relation" field must contain ONLY the relation predicate/preposition WITHOUT the object name!
   - WRONG: "perched on buoy" (includes object name "buoy")
   - CORRECT: "perched on" (pure relation only)
   - WRONG: "beside the net" (includes object name "net")
   - CORRECT: "beside" (pure relation only)
   - WRONG: "flying over ocean" (includes object name)
   - CORRECT: "flying over" (pure relation only)
   
   The object name is already stored via object_id, so DO NOT repeat it in the relation field!

6. CRITICAL - RELATION CONSISTENCY: All relations must describe the SAME MOMENT in time and be physically compatible:
   - STATIC relations (perched on, standing on, sitting on, resting on, lying on) = subject is NOT moving
   - DYNAMIC relations (flying over, soaring above, swimming past, gliding over) = subject is IN MOTION
   - NEUTRAL relations (near, beside, in front of, behind, with, surrounded by) = compatible with both
   
   RULE: If you use a STATIC relation, ALL other relations must be STATIC or NEUTRAL (no DYNAMIC)
   RULE: If you use a DYNAMIC relation, ALL other relations must be DYNAMIC or NEUTRAL (no STATIC)
   
   WRONG: "perched on" + "soaring above" (static + dynamic = CONFLICT!)
   CORRECT: "perched on" + "near" (static + neutral = OK)
   CORRECT: "flying over" + "soaring above" (dynamic + dynamic = OK)

7. CRITICAL: The caption MUST strictly use the EXACT same relations you defined in the scene graph
   - Each relation must appear in the caption using the same verb phrase
   - The subject should be the actor of each relation

8. Keep the caption concise but descriptive (1-2 sentences)

9. SUPERCLASS: If a superclass is provided (e.g., "bird"), you MUST include it right after the subject name in the caption.
   - Example: If subject is "Albatross" and superclass is "bird", write "A Albatross bird" NOT just "A Albatross"
   - This helps clarify what type of entity the subject is

Example:
Subject: Albatross, Objects: [dock, ocean]
CORRECT output:
{
    "relations": [
        {"subject_id": 0, "object_id": 1, "relation": "perched on"},
        {"subject_id": 0, "object_id": 2, "relation": "beside"}
    ],
    "caption": "An Albatross is perched on a dock beside the ocean."
}
WRONG: {"relation": "perched on dock"} - DO NOT include object name in relation!

Example with superclass:
Subject: Albatross, Superclass: bird, Objects: [dock, ocean]
CORRECT: caption="An Albatross bird is perched on a dock beside the ocean."

Output format (JSON):
{
    "relations": [
        {"subject_id": 0, "object_id": 1, "relation": "pure_relation_without_object_name"},
        ...
    ],
    "caption": "A natural description using the EXACT relations defined above."
}

Return ONLY the JSON object, no extra text.
"""

    def __init__(
        self,
        min_objects: int = 1,
        max_objects: int = 3,
        max_retries: int = 5,
        superclass: Optional[str] = None,
        sampling_mode: str = "mixed",  # "mixed" or "single_category"
        objects_per_category: Optional[int] = None,  # For single_category mode
        sampling_category: Optional[str] = None  # For single_category mode
    ):
        """
        Initialize the generator.
        
        Args:
            min_objects: Minimum number of objects to sample per scene
            max_objects: Maximum number of objects to sample per scene
            max_retries: Maximum number of retries for validation failures
            superclass: Optional superclass name to append after subject in captions
                       (e.g., "bird" for bird datasets, so "Albatross" becomes "Albatross bird")
            sampling_mode: Sampling strategy - "mixed" (random from all categories) or 
                          "single_category" (from one category only)
            objects_per_category: Number of objects to sample per category in single_category mode
                                (defaults to a random value within min/max bounds)
            sampling_category: Specific category to sample from in single_category mode
                               (defaults to random non-empty category)
        """
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.max_retries = max_retries
        self.superclass = superclass
        self.sampling_mode = sampling_mode
        self.objects_per_category = objects_per_category
        self.sampling_category = sampling_category
        
        if sampling_mode not in ["mixed", "single_category"]:
            raise ValueError(f"sampling_mode must be 'mixed' or 'single_category', got '{sampling_mode}'")
        if sampling_category and sampling_category not in self.ALLOWED_CATEGORIES:
            raise ValueError(
                "sampling_category must be one of "
                f"{', '.join(self.ALLOWED_CATEGORIES)}, got '{sampling_category}'"
            )
        
        objects_per_category_log = (
            self.objects_per_category
            if self.objects_per_category is not None
            else "min/max"
        )
        logger.info(
            f"Initialized SceneGraphGenerator with "
            f"min_objects={min_objects}, max_objects={max_objects}, "
            f"superclass={superclass}, sampling_mode={sampling_mode}, "
            f"objects_per_category={objects_per_category_log}, "
            f"sampling_category={self.sampling_category}"
        )
    
    def load_coobjects(self, input_path: str) -> List[CoObjectItem]:
        """
        Load co-objects from a JSON file.
        
        Args:
            input_path: Path to the input JSON file
            
        Returns:
            List of CoObjectItem objects
        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        items = [CoObjectItem.from_dict(item) for item in data]
        logger.info(f"Loaded {len(items)} co-object items from {input_path}")
        
        return items
    
    def sample_objects(
        self,
        item: "CoObjectItem",
        min_count: Optional[int] = None,
        max_count: Optional[int] = None
    ) -> List[str]:
        """
        Sample objects from the co-object item based on sampling mode.
        
        Args:
            item: CoObjectItem with three categories
            min_count: Minimum number to sample (defaults to self.min_objects)
            max_count: Maximum number to sample (defaults to self.max_objects)
            
        Returns:
            List of sampled objects
        """
        min_count = min_count or self.min_objects
        max_count = max_count or self.max_objects
        
        if self.sampling_mode == "mixed":
            # Random sampling from all three categories (1-3 objects total)
            return self._sample_mixed_categories(item, min_count, max_count)
        else:  # single_category
            # Sample from one category (specified or randomly selected)
            return self._sample_single_category(
                item,
                min_count,
                max_count,
                self.objects_per_category,
                self.sampling_category
            )
    
    def _sample_mixed_categories(
        self,
        item: "CoObjectItem",
        min_count: int,
        max_count: int
    ) -> List[str]:
        """
        Sample objects randomly from all three categories (mixed mode).
        
        Args:
            item: CoObjectItem with categories
            min_count: Minimum number of objects to sample
            max_count: Maximum number of objects to sample
            
        Returns:
            List of sampled objects
        """
        # Combine all categories into a pool
        all_objects = []
        all_objects.extend(item.semantically_associated)
        all_objects.extend(item.compatible_non_typical)
        all_objects.extend(item.contextually_contrastive)
        
        if not all_objects:
            return []
        
        # Random number of objects to sample (1-3)
        max_possible = min(max_count, len(all_objects))
        min_possible = min(min_count, max_possible)
        n_sample = random.randint(min_possible, max_possible)
        
        return random.sample(all_objects, n_sample)
    
    def _sample_single_category(
        self,
        item: "CoObjectItem",
        min_count: int,
        max_count: int,
        count: Optional[int] = None,
        category: Optional[str] = None
    ) -> List[str]:
        """
        Sample objects from a single randomly selected category.
        
        Args:
            item: CoObjectItem with categories
            min_count: Minimum number of objects to sample
            max_count: Maximum number of objects to sample
            count: Fixed number of objects to sample (optional)
            category: Specific category to sample from (optional)
            
        Returns:
            List of sampled objects
        """
        if category:
            category_objects = self._get_category_objects(item, category)
            if not category_objects:
                return []
            n_sample = self._resolve_sample_count(
                category_objects,
                min_count,
                max_count,
                count
            )
            return random.sample(category_objects, n_sample)

        # Collect non-empty categories
        categories = []
        if item.semantically_associated:
            categories.append(("semantically_associated", item.semantically_associated))
        if item.compatible_non_typical:
            categories.append(("compatible_non_typical", item.compatible_non_typical))
        if item.contextually_contrastive:
            categories.append(("contextually_contrastive", item.contextually_contrastive))
        
        if not categories:
            return []
        
        # Randomly select one category
        category_name, category_objects = random.choice(categories)
        
        # Sample from the selected category
        n_sample = self._resolve_sample_count(
            category_objects,
            min_count,
            max_count,
            count
        )
        return random.sample(category_objects, n_sample)

    def _get_category_objects(self, item: "CoObjectItem", category: str) -> List[str]:
        """Get objects for a specific category from a co-object item."""
        if category == "semantically_associated":
            return item.semantically_associated
        if category == "compatible_non_typical":
            return item.compatible_non_typical
        if category == "contextually_contrastive":
            return item.contextually_contrastive
        return []

    def _resolve_sample_count(
        self,
        category_objects: List[str],
        min_count: int,
        max_count: int,
        count: Optional[int]
    ) -> int:
        """Resolve the number of objects to sample within bounds."""
        if not category_objects:
            return 0
        if count is not None:
            return min(count, len(category_objects))
        max_possible = min(max_count, len(category_objects))
        min_possible = min(min_count, max_possible)
        return random.randint(min_possible, max_possible)
    
    def create_generation_tasks(
        self,
        coobjects: List[CoObjectItem],
        samples_per_subject: int = 10
    ) -> List[GenerationTask]:
        """
        Create generation tasks for all subjects.
        
        Args:
            coobjects: List of CoObjectItem objects
            samples_per_subject: Number of samples to generate per subject
            
        Returns:
            List of GenerationTask objects with sequential task_ids
        """
        tasks = []
        task_id = 0
        
        for item in coobjects:
            for _ in range(samples_per_subject):
                sampled_objects = self.sample_objects(item)
                task = GenerationTask(
                    task_id=task_id,
                    subject=item.subject,
                    sampled_objects=sampled_objects,
                    label=item.label,
                    label_name=item.label_name
                )
                tasks.append(task)
                task_id += 1
        
        logger.info(f"Created {len(tasks)} generation tasks")
        return tasks
    
    def build_prompt(self, subject: str, objects: List[str]) -> str:
        """
        Build the user prompt for LLM.
        
        Args:
            subject: The subject name
            objects: List of object names
            
        Returns:
            The formatted user prompt
        """
        objects_numbered = "\n".join([f"  {i+1}. {obj}" for i, obj in enumerate(objects)])
        
        # Add superclass instruction if specified
        superclass_instruction = ""
        if self.superclass:
            superclass_instruction = f"\nSuperclass: {self.superclass} (IMPORTANT: In the caption, refer to the subject as \"{subject} {self.superclass}\")"
        
        return f"""Subject (id=0): {subject}{superclass_instruction}

Objects:
{objects_numbered}

Generate the scene graph relations and caption for this scene.
Return ONLY a valid JSON object."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing underscores and unwanted special characters.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        # Replace underscores with spaces
        cleaned = text.replace("_", " ")
        
        # Remove special characters but keep basic punctuation for captions
        # Keep: letters, numbers, spaces, and basic punctuation (.,!?'-:;)
        cleaned = re.sub(r'[^\w\s.,!?\'\-:;]', '', cleaned)
        
        # Remove multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    @staticmethod
    def clean_relation(relation: str, object_name: Optional[str] = None) -> str:
        """
        Clean relation text by removing underscores, special characters, and object name.
        
        Args:
            relation: The relation text to clean
            object_name: Optional object name to strip from the relation
            
        Returns:
            Cleaned relation text (pure relation without object name)
        """
        # Replace underscores with spaces
        cleaned = relation.replace("_", " ")
        
        # For relations, only keep letters, numbers, and spaces
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        
        # Remove multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        cleaned = cleaned.strip().lower()
        
        # If object_name is provided, strip it from the relation
        if object_name:
            obj_lower = object_name.lower().strip()
            # Remove object name if it appears at the end of relation
            # Handle cases like "perched on buoy" -> "perched on"
            # Also handle "perched on the buoy" -> "perched on"
            
            # Pattern: relation ends with (the/a/an)? object_name
            patterns = [
                rf'\s+(?:the\s+|a\s+|an\s+)?{re.escape(obj_lower)}\s*$',  # "on the buoy"
                rf'\s+{re.escape(obj_lower)}\s*$',  # "on buoy"
            ]
            
            for pattern in patterns:
                cleaned = re.sub(pattern, '', cleaned)
            
            # Also handle multi-word object names
            obj_words = obj_lower.split()
            if len(obj_words) > 1:
                # Try to match any suffix that contains all object words
                for word in obj_words:
                    # Remove trailing object word with optional article
                    cleaned = re.sub(rf'\s+(?:the\s+|a\s+|an\s+)?{re.escape(word)}\s*$', '', cleaned)
        
        return cleaned.strip()
    
    @staticmethod
    def has_invalid_characters(text: str) -> bool:
        """
        Check if text contains invalid characters like underscores or excessive special chars.
        
        Args:
            text: The text to check
            
        Returns:
            True if text contains invalid characters
        """
        # Check for underscores
        if "_" in text:
            return True
        
        # Check for excessive special characters (more than 20% of text)
        special_count = len(re.findall(r'[^\w\s.,!?\'\-:;]', text))
        if len(text) > 0 and special_count / len(text) > 0.2:
            return True
        
        return False

    def create_validator(
        self,
        subject: str,
        objects: List[str]
    ) -> Callable[[str], Optional[Dict[str, Any]]]:
        """
        Create a validation function for LLM responses.
        
        Args:
            subject: The subject name
            objects: List of object names
            
        Returns:
            A validation function that returns parsed data or None if invalid
        """
        expected_object_ids = set(range(1, len(objects) + 1))
        
        def validate(response: str) -> Optional[Dict[str, Any]]:
            """
            Validate and parse the LLM response.
            
            Args:
                response: The raw LLM response string
                
            Returns:
                Parsed data dict or None if validation fails
            """
            try:
                # Try to extract JSON object from response
                # Handle cases where response might have extra text
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if not json_match:
                    logger.warning(f"No JSON object found in response: {response[:100]}...")
                    return None
                
                json_str = json_match.group()
                data = json.loads(json_str)
                
                # Validate required fields
                if "relations" not in data or "caption" not in data:
                    logger.warning("Missing required fields: relations or caption")
                    return None
                
                relations = data["relations"]
                caption = data["caption"]
                
                # Validate relations is a list
                if not isinstance(relations, list):
                    logger.warning("Relations is not a list")
                    return None
                
                # Validate caption is a string
                if not isinstance(caption, str) or not caption.strip():
                    logger.warning("Caption is not a valid string")
                    return None
                
                # Clean caption - remove underscores and special characters
                caption = SceneGraphGenerator.clean_text(caption)
                
                # Check if caption still has invalid characters after cleaning
                if not caption or len(caption) < 10:
                    logger.warning("Caption too short after cleaning")
                    return None
                
                # Validate each relation
                validated_relations = []
                seen_object_ids = set()
                
                for rel in relations:
                    if not isinstance(rel, dict):
                        continue
                    
                    subject_id = rel.get("subject_id")
                    object_id = rel.get("object_id")
                    relation_text = rel.get("relation")
                    
                    # Validate subject_id is 0
                    if subject_id != 0:
                        continue
                    
                    # Validate object_id is in expected range
                    if object_id not in expected_object_ids:
                        continue
                    
                    # Validate relation is a non-empty string
                    if not isinstance(relation_text, str) or not relation_text.strip():
                        continue
                    
                    # Get the object name for this object_id to strip from relation
                    object_name = objects[object_id - 1] if object_id <= len(objects) else None
                    
                    # Clean relation text - remove underscores, special characters, and object name
                    relation_text = SceneGraphGenerator.clean_relation(relation_text, object_name)
                    
                    # Validate cleaned relation is not empty
                    if not relation_text:
                        logger.warning(f"Relation text empty after cleaning for object_id {object_id}")
                        continue
                    
                    # Skip duplicate object_ids
                    if object_id in seen_object_ids:
                        continue
                    
                    seen_object_ids.add(object_id)
                    validated_relations.append({
                        "subject_id": 0,
                        "object_id": object_id,
                        "relation": relation_text
                    })
                
                # Ensure we have relations for all objects
                if len(validated_relations) != len(objects):
                    logger.warning(
                        f"Not enough valid relations: {len(validated_relations)} != {len(objects)}"
                    )
                    return None
                
                return {
                    "relations": validated_relations,
                    "caption": caption
                }
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
                return None
            except Exception as e:
                logger.warning(f"Validation error: {e}")
                return None
        
        return validate
    
    def build_scene_graph_item(
        self,
        task: GenerationTask,
        llm_response: Dict[str, Any],
        item_id: int
    ) -> SceneGraphItem:
        """
        Build a SceneGraphItem from task and LLM response.
        
        Args:
            task: The generation task
            llm_response: Validated LLM response dict
            item_id: The ID for this item
            
        Returns:
            SceneGraphItem object
        """
        # Create subject entity
        subject_entity = ObjectEntity(id=0, name=task.subject)
        
        # Create object entities
        object_entities = [
            ObjectEntity(id=i+1, name=obj)
            for i, obj in enumerate(task.sampled_objects)
        ]
        
        # Create relations
        relations = [
            Relation(
                subject_id=rel["subject_id"],
                object_id=rel["object_id"],
                relation=rel["relation"]
            )
            for rel in llm_response["relations"]
        ]
        
        # Create scene graph
        scene_graph = SceneGraph(
            id=item_id,
            subject=subject_entity,
            objects=object_entities,
            relations=relations
        )
        
        # Create and return item with label info preserved
        return SceneGraphItem(
            id=item_id,
            label=task.label,
            label_name=task.label_name,
            scene_graph=scene_graph,
            caption=llm_response["caption"]
        )
    
    def prepare_prompts_with_tasks(
        self,
        tasks: List[GenerationTask]
    ) -> List[Tuple[int, str, GenerationTask]]:
        """
        Prepare prompts with task information for batch processing.
        
        Args:
            tasks: List of GenerationTask objects
            
        Returns:
            List of tuples (task_id, prompt, task)
        """
        prompts = []
        for task in tasks:
            prompt = self.build_prompt(task.subject, task.sampled_objects)
            prompts.append((task.task_id, prompt, task))
        
        return prompts
    
    def save_results(
        self,
        results: List[SceneGraphItem],
        output_path: str,
        dataset_name: str = "synthetic"
    ) -> None:
        """
        Save results to a JSON file.
        
        Args:
            results: List of SceneGraphItem objects
            output_path: Path to the output JSON file
            dataset_name: Name of the dataset
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset = SyntheticDataset(
            dataset=dataset_name,
            total=len(results),
            results=results
        )
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(results)} results to {output_path}")
    
    def append_result(
        self,
        result: SceneGraphItem,
        output_path: str
    ) -> None:
        """
        Append a single result to the output file (for streaming).
        
        This is used for real-time output during generation.
        
        Args:
            result: The SceneGraphItem to append
            output_path: Path to the output JSON file
        """
        path = Path(output_path)
        
        # If file doesn't exist or is empty, create new structure
        if not path.exists() or path.stat().st_size == 0:
            dataset = SyntheticDataset(
                dataset="synthetic",
                total=1,
                results=[result]
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dataset.to_dict(), f, indent=2, ensure_ascii=False)
        else:
            # Read existing data
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Append new result
            data["results"].append(result.to_dict())
            data["total"] = len(data["results"])
            
            # Write back
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for LLM."""
        return self.SYSTEM_PROMPT

