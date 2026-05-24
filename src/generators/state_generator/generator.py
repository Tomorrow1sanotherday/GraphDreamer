"""
Subject State Generator Module

This module provides functionality to generate visual states (pose, activity, condition)
for given subjects (labels) using LLM (DeepSeek).

States are single words describing how the subject can appear in an image, e.g.:
flying, perching, standing, feeding, preening, resting, walking.
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

logger = logging.getLogger(__name__)


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
class StateResult:
    """Represents the state result for a single subject."""
    label: int
    label_name: str
    subject: str
    states: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "label": self.label,
            "label_name": self.label_name,
            "subject": self.subject,
            "states": self.states
        }


class SubjectStateGenerator:
    """
    Generator for subject states (pose, activity, condition) using LLM.

    This class handles:
    - Loading labels from input JSON
    - Generating prompts for LLM
    - Parsing and validating LLM responses
    - Saving results to output JSON
    """

    BASE_RULES = """One word per state. Lowercase. No duplicates. Return ONLY a JSON array of strings."""

    SYSTEM_PROMPT = """For the given subject (a specific kind within a superclass), list one-word states that are common for this category (pose/action/condition). {base_rules}"""

    def __init__(
        self,
        n_states_per_subject: int = 20,
        max_retries: int = 10
    ):
        """
        Initialize the generator.

        Args:
            n_states_per_subject: Number of states to generate per subject
            max_retries: Maximum number of retries for validation failures
        """
        self.n_states_per_subject = n_states_per_subject
        self.max_retries = max_retries

        logger.info(
            f"Initialized SubjectStateGenerator with n_states_per_subject={n_states_per_subject}"
        )

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
        superclass: str = "Bird"
    ) -> str:
        """
        Build the user prompt for LLM.

        Args:
            subject: The subject name (with spaces, not underscores)
            superclass: The broad category of the subject (e.g., Bird, Vehicle)

        Returns:
            The formatted user prompt
        """
        request_count = self.n_states_per_subject + 5

        prompt = f'''Superclass: "{superclass}"
Subject: "{subject}"

List exactly {request_count} one-word states that are common for this category. Return ONLY a JSON array.'''

        return prompt

    def create_validator(
        self,
        subject: str,
        superclass: str = "Bird"
    ) -> Callable[[str], Optional[List[str]]]:
        """
        Create a validation function for LLM responses.

        Args:
            subject: The subject name for validation
            superclass: The broad category of the subject

        Returns:
            A validation function that returns parsed states list or None if invalid
        """
        def validate(response: str) -> Optional[List[str]]:
            """
            Validate and parse the LLM response.

            Args:
                response: The raw LLM response string

            Returns:
                List of valid state strings or None if validation fails
            """
            try:
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if not json_match:
                    logger.warning(f"No JSON array found in response: {response[:100]}...")
                    return None

                json_str = json_match.group()
                states = json.loads(json_str)

                if not isinstance(states, list):
                    logger.warning("Response is not a list")
                    return None

                if not all(isinstance(s, str) for s in states):
                    logger.warning("Not all items are strings")
                    return None

                # Clean: one word only — strip, lowercase, take first word if phrase
                cleaned = []
                for s in states:
                    t = s.strip().lower()
                    if not t:
                        continue
                    word = t.split()[0] if t.split() else t
                    if word:
                        cleaned.append(word)

                # Remove duplicates while preserving order
                seen = set()
                unique = []
                for s in cleaned:
                    if s not in seen:
                        seen.add(s)
                        unique.append(s)

                # Keep only single-word states (already enforced above)
                filtered = [s for s in unique if len(s.split()) == 1 and len(s) <= 30]

                if len(filtered) < self.n_states_per_subject:
                    logger.warning(
                        f"Not enough states for '{subject}' after filtering: "
                        f"{len(filtered)} < {self.n_states_per_subject}"
                    )
                    return None

                return filtered[:self.n_states_per_subject]

            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
                return None
            except Exception as e:
                logger.warning(f"Validation error: {e}")
                return None

        return validate

    def get_system_prompt(self) -> str:
        """Get the system prompt for LLM."""
        return self.SYSTEM_PROMPT.replace("{base_rules}", self.BASE_RULES)

    def save_results(
        self,
        results: List[StateResult],
        output_path: str
    ) -> None:
        """
        Save results to a JSON file.

        Args:
            results: List of StateResult objects
            output_path: Path to the output JSON file
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        data = [r.to_dict() for r in results]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        logger.info(f"Saved {len(results)} results to {output_path}")
