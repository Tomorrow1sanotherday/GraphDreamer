"""
Attribute Generator Module

This module provides functionality to generate distinguishing attributes
for given subjects using LLM.

Attributes are generated to distinguish the subject from other members
of the same superclass.
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .data_types import CoObjectItem, AttributeResult

logger = logging.getLogger(__name__)


class AttributeGenerator:
    """
    Generator for distinguishing attributes using LLM.
    
    This class handles:
    - Loading co-object items from input JSON
    - Generating prompts for LLM to get distinguishing attributes
    - Parsing and validating LLM responses
    - Saving results to output JSON
    """
    
    # System prompt for attribute generation
    SYSTEM_PROMPT = """You are an expert in identifying distinguishing attributes for classification tasks.

Your task: Generate distinguishing attributes that help differentiate a specific subject from other members of the same superclass.

Guidelines:
1. Focus on attributes that are VISIBLE and can be observed in images
2. Attributes should be distinctive characteristics that set this subject apart from other members of the superclass
3. Each attribute should be a key-value pair where:
   - Key: The attribute name - MUST be SPECIFIC, not generic (e.g., "head color", "wing pattern", "beak shape", NOT "color", "size", "pattern")
   - Value: The specific value for this subject - MUST be BRIEF and CONCISE (1-3 words maximum)
   - CRITICAL: The attribute value should NOT repeat concepts from the attribute name
     * If attribute name is "wing pattern", value should NOT contain "wing" (e.g., use "black bars" not "black wing bars")
     * If attribute name is "beak shape", value should NOT contain "beak" (e.g., use "curved" not "curved beak")
     * If attribute name is "head color", value should be just the color (e.g., "yellow", "black")
4. Attributes should be concrete and specific, not abstract or vague
5. Keep values SHORT - use single words or very brief phrases (2-3 words max)
6. EXCLUDE attributes that are difficult to quantify or generate in text-to-image models:
   - DO NOT use: "size", "length", "weight", "proportion", "scale", "dimension"
   - DO NOT use generic attributes like "color" - use specific body part colors instead (e.g., "head color", "wing color", "body color")
7. Prefer specific visual features: colors of specific body parts, patterns, shapes, markings, textures
8. Return ONLY a JSON array of objects, where each object is a dictionary with a single key-value pair

Example format:
[
  {"head color": "yellow"},
  {"wing pattern": "black bars"},
  {"beak shape": "curved"},
  {"tail marking": "white spots"}
]

Return ONLY the JSON array, no additional text or explanation."""

    def __init__(
        self,
        n_attributes: int = 5,
        max_retries: int = 10
    ):
        """
        Initialize the generator.
        
        Args:
            n_attributes: Number of distinguishing attributes to generate per subject
            max_retries: Maximum number of retries for validation failures
        """
        self.n_attributes = n_attributes
        self.max_retries = max_retries
        
        logger.info(f"Initialized AttributeGenerator with n_attributes={n_attributes}")
    
    def load_coobjects(self, input_path: str) -> List[CoObjectItem]:
        """
        Load co-object items from a JSON file.
        
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
    
    def build_prompt(
        self, 
        subject: str, 
        superclass: str = "Bird",
        n: int = None
    ) -> str:
        """
        Build the user prompt for LLM.
        
        Args:
            subject: The subject name (with spaces, not underscores)
            superclass: The broad category of the subject (e.g., Bird, Vehicle)
            n: Number of attributes to generate (defaults to self.n_attributes)
            
        Returns:
            The formatted user prompt
        """
        n = n or self.n_attributes
        
        prompt = f'''Subject: "{subject}"
Superclass: "{superclass}"

Generate exactly {n} distinguishing attributes that help differentiate this {subject} from other members of the {superclass} superclass.

Requirements:
- Focus on visible, observable characteristics that are distinctive for this specific subject
- Keep attribute values BRIEF and CONCISE (1-3 words maximum)
- Attribute values must NOT repeat concepts from the attribute name:
  * If attribute name is "wing pattern", value should NOT contain "wing"
  * If attribute name is "beak shape", value should NOT contain "beak"
- Use SPECIFIC attribute names, not generic ones:
  * Use "head color", "wing color", "body color" instead of generic "color"
  * Use "wing pattern", "tail pattern" instead of generic "pattern"
  * Use "beak shape", "tail shape" instead of generic "shape"
- EXCLUDE attributes that are difficult to quantify:
  * DO NOT use: "size", "length", "weight", "proportion", "scale", "dimension"
  * These are hard to quantify and difficult for text-to-image models to generate accurately

Return ONLY a JSON array of objects, where each object contains a single key-value pair representing one attribute.
Example format: [{{"head color": "yellow"}}, {{"wing pattern": "black bars"}}, {{"beak shape": "curved"}}]'''
        
        return prompt
    
    def create_validator(
        self, 
        n: int = None
    ) -> Callable[[str], Optional[List[Dict[str, str]]]]:
        """
        Create a validation function for LLM responses.
        
        Args:
            n: Expected number of attributes (defaults to self.n_attributes)
            
        Returns:
            A validation function that returns parsed attributes list or None if invalid
        """
        n = n or self.n_attributes
        
        def validate(response: str) -> Optional[List[Dict[str, str]]]:
            """
            Validate and parse the LLM response.
            
            Args:
                response: The raw LLM response string
                
            Returns:
                List of attribute dictionaries or None if validation fails
            """
            try:
                # Try to extract JSON array from response
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if not json_match:
                    logger.warning(f"No JSON array found in response: {response[:100]}...")
                    return None
                
                json_str = json_match.group()
                attributes = json.loads(json_str)
                
                # Validate it's a list
                if not isinstance(attributes, list):
                    logger.warning("Response is not a list")
                    return None
                
                # Validate all items are dictionaries
                if not all(isinstance(attr, dict) for attr in attributes):
                    logger.warning("Not all items are dictionaries")
                    return None
                
                # Validate each dictionary has exactly one key-value pair
                validated_attributes = []
                for attr in attributes:
                    if len(attr) != 1:
                        logger.warning(f"Attribute dictionary has {len(attr)} keys, expected 1: {attr}")
                        continue
                    
                    # Ensure both key and value are strings
                    key, value = next(iter(attr.items()))
                    if not isinstance(key, str) or not isinstance(value, str):
                        logger.warning(f"Attribute key or value is not a string: {attr}")
                        continue
                    
                    # Create a new dict with the single key-value pair
                    validated_attributes.append({key: value})
                
                # Check we have enough attributes
                if len(validated_attributes) < n:
                    logger.warning(f"Not enough attributes after validation: {len(validated_attributes)} < {n}")
                    return None
                
                # Take only the required number
                return validated_attributes[:n]
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
                return None
            except Exception as e:
                logger.warning(f"Validation error: {e}")
                return None
        
        return validate
    
    def save_results(
        self,
        results: List[AttributeResult],
        output_path: str
    ) -> None:
        """
        Save results to a JSON file.
        
        Args:
            results: List of AttributeResult objects
            output_path: Path to the output JSON file
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data = [result.to_dict() for result in results]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Saved {len(results)} results to {output_path}")
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for LLM.
        
        Returns:
            The system prompt
        """
        return self.SYSTEM_PROMPT

