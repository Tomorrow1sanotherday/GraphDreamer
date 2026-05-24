#!/usr/bin/env python3
"""
Script to add attributes to scene graphs and update captions.

This script reads scene graphs from an input JSON file, looks up attributes
from the coobjects_with_attributes file, adds attributes to scene graphs,
and updates captions with attribute information.

Usage:
    python run_generate_scene_graph_wa.py \
        --scene_graphs_path /path/to/scene_graphs.json \
        --attributes_path /path/to/coobjects_with_attributes.json \
        --output_path /path/to/output.json
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def load_json(file_path: str) -> Any:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str) -> None:
    """Save data to JSON file."""
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved results to {file_path}")


def build_attribute_index(attributes_data: List[Dict]) -> Dict[str, Dict]:
    """
    Build an index mapping label_name or subject name to attributes.
    
    Args:
        attributes_data: List of attribute entries from coobjects_with_attributes.json
        
    Returns:
        Dictionary mapping label_name -> attributes entry
    """
    index = {}
    
    for entry in attributes_data:
        label_name = entry.get('label_name')
        subject = entry.get('subject', '')
        
        if label_name:
            index[label_name] = entry
        
        # Also index by subject name (normalized)
        if subject:
            # Normalize subject name: "Black footed Albatross" -> "Black_footed_Albatross"
            normalized_subject = subject.replace(' ', '_')
            if normalized_subject not in index:
                index[normalized_subject] = entry
    
    logger.info(f"Built attribute index with {len(index)} entries")
    return index


def format_attributes_string(attributes: List[Dict]) -> str:
    """
    Format attributes list into a string like "dark brown body color, white patch head marking".
    
    Args:
        attributes: List of attribute dictionaries, e.g., [{"body color": "dark brown"}, ...]
        
    Returns:
        Formatted string
    """
    if not attributes:
        return ""
    
    parts = []
    for attr_dict in attributes:
        # Each dict has one key-value pair
        for concept, value in attr_dict.items():
            # Format: "value concept"
            parts.append(f"{value} {concept}")
    
    return ", ".join(parts)


def update_caption_with_attributes(caption: str, attributes_str: str) -> str:
    """
    Update caption by inserting attributes after the subject name.
    
    Example:
        Input: "A Black footed Albatross bird is perched on the rigging."
        Attributes: "dark brown body color, white patch head marking"
        Output: "A Black footed Albatross bird which has dark brown body color, white patch head marking is perched on the rigging."
    
    Args:
        caption: Original caption
        attributes_str: Formatted attributes string
        
    Returns:
        Updated caption
    """
    if not attributes_str:
        return caption
    
    # Pattern to match "A [Subject] bird" - most common pattern
    # Match: "A" + subject name (capitalized words) + "bird"
    pattern1 = r'(A\s+[A-Z][A-Za-z\s]+?\s+bird)'
    match = re.search(pattern1, caption, re.IGNORECASE)
    if match:
        subject_phrase = match.group(1)
        # Insert "which has [attributes]" after "bird"
        updated = caption.replace(
            subject_phrase,
            f"{subject_phrase} which has {attributes_str}",
            1
        )
        return updated
    
    # Pattern to match "A [Subject]" followed by "is" or other verb
    pattern2 = r'(A\s+[A-Z][A-Za-z\s]+?)(\s+is\s+)'
    match = re.search(pattern2, caption, re.IGNORECASE)
    if match:
        subject_phrase = match.group(1)
        verb_part = match.group(2)
        # Insert "which has [attributes]" before the verb
        updated = caption.replace(
            subject_phrase + verb_part,
            f"{subject_phrase} which has {attributes_str}{verb_part}",
            1
        )
        return updated
    
    # Fallback: try to find "A [Subject]" anywhere
    pattern3 = r'(A\s+[A-Z][A-Za-z\s]+?)(\s+[a-z])'
    match = re.search(pattern3, caption, re.IGNORECASE)
    if match:
        subject_phrase = match.group(1)
        next_part = match.group(2)
        # Insert "which has [attributes]" after subject
        updated = caption.replace(
            subject_phrase + next_part,
            f"{subject_phrase} which has {attributes_str}{next_part}",
            1
        )
        return updated
    
    # Last fallback: prepend attributes
    logger.warning(f"Could not find subject pattern in caption: {caption}")
    return f"A bird which has {attributes_str} {caption.lower()}"


def add_attributes_to_scene_graphs(
    scene_graphs_data: Dict,
    attributes_index: Dict[str, Dict]
) -> Dict:
    """
    Add attributes to scene graphs and update captions.
    
    Args:
        scene_graphs_data: Scene graphs data dictionary
        attributes_index: Index mapping label_name to attributes
        
    Returns:
        Updated scene graphs data
    """
    results = scene_graphs_data.get('results', [])
    total = len(results)
    updated_count = 0
    missing_count = 0
    
    logger.info(f"Processing {total} scene graphs...")
    
    for item in results:
        label_name = item.get('label_name')
        scene_graph = item.get('scene_graph', {})
        subject_name = scene_graph.get('subject', {}).get('name', '')
        
        # Try to find attributes by label_name first
        attributes_entry = None
        if label_name and label_name in attributes_index:
            attributes_entry = attributes_index[label_name]
        elif subject_name:
            # Try normalized subject name
            normalized_subject = subject_name.replace(' ', '_')
            if normalized_subject in attributes_index:
                attributes_entry = attributes_index[normalized_subject]
        
        if attributes_entry:
            attributes = attributes_entry.get('attributes', [])
            
            # Add attributes to scene_graph
            if 'attributes' not in scene_graph:
                scene_graph['attributes'] = []
            scene_graph['attributes'] = attributes
            
            # Update caption
            if attributes:
                attributes_str = format_attributes_string(attributes)
                original_caption = item.get('caption', '')
                updated_caption = update_caption_with_attributes(original_caption, attributes_str)
                item['caption'] = updated_caption
            
            updated_count += 1
        else:
            missing_count += 1
            logger.warning(f"No attributes found for label_name={label_name}, subject={subject_name}")
    
    logger.info(f"Updated {updated_count} scene graphs with attributes")
    if missing_count > 0:
        logger.warning(f"Could not find attributes for {missing_count} scene graphs")
    
    return scene_graphs_data


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Add attributes to scene graphs and update captions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--scene_graphs_path',
        type=str,
        required=True,
        help='Path to input scene graphs JSON file'
    )
    
    parser.add_argument(
        '--attributes_path',
        type=str,
        required=True,
        help='Path to coobjects_with_attributes JSON file'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to output JSON file or directory (if directory, auto-generate filename with "wa" suffix; default: same directory as input with "wa" suffix)'
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output_path is None:
        input_path = Path(args.scene_graphs_path)
        # Add "wa" before the extension
        output_path = input_path.parent / f"{input_path.stem}wa{input_path.suffix}"
    else:
        # Check if output_path is a directory before creating Path object
        # Check original string first (for trailing slash)
        output_path_str = args.output_path
        output_path = Path(output_path_str)
        
        # Check if output_path is a directory
        # Conditions: exists and is directory, OR ends with /, OR has no extension
        is_directory = (
            (output_path.exists() and output_path.is_dir()) or
            output_path_str.endswith('/') or
            (output_path.suffix == '' and output_path.name != '')
        )
        
        if is_directory:
            # It's a directory, generate filename from input
            input_path = Path(args.scene_graphs_path)
            output_path = output_path / f"{input_path.stem}wa{input_path.suffix}"
    
    logger.info("=" * 70)
    logger.info("Scene Graph Attributes Adder")
    logger.info("=" * 70)
    logger.info(f"Input scene graphs: {args.scene_graphs_path}")
    logger.info(f"Input attributes: {args.attributes_path}")
    logger.info(f"Output path: {output_path}")
    
    # Load data
    logger.info("\nLoading scene graphs...")
    scene_graphs_data = load_json(args.scene_graphs_path)
    logger.info(f"Loaded {len(scene_graphs_data.get('results', []))} scene graphs")
    
    logger.info("\nLoading attributes...")
    attributes_data = load_json(args.attributes_path)
    logger.info(f"Loaded {len(attributes_data)} attribute entries")
    
    # Build attribute index
    logger.info("\nBuilding attribute index...")
    attributes_index = build_attribute_index(attributes_data)
    
    # Add attributes to scene graphs
    logger.info("\nAdding attributes to scene graphs...")
    updated_data = add_attributes_to_scene_graphs(scene_graphs_data, attributes_index)
    
    # Save results
    logger.info(f"\nSaving results to {output_path}...")
    save_json(updated_data, str(output_path))
    
    logger.info("\n" + "=" * 70)
    logger.info("Done!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

