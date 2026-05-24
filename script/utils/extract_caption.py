#!/usr/bin/env python3
"""
Script to extract captions from syn_train_index JSON file.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def extract_captions(json_path: str) -> List[Dict]:
    """
    Extract captions from the JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        List of dictionaries containing id and caption
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    captions = []
    for item in data.get('results', []):
        captions.append({
            'id': item.get('id'),
            'caption': item.get('caption', ''),
            'label_name': item.get('label_name', ''),
            'image_path': item.get('image_path', '')
        })
    
    return captions


def save_to_text(captions: List[Dict], output_path: str):
    """Save captions to a text file, one per line."""
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in captions:
            f.write(item['caption'] + '\n')


def save_to_json(captions: List[Dict], output_path: str):
    """Save captions to a JSON file."""
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)


def save_to_csv(captions: List[Dict], output_path: str):
    """Save captions to a CSV file."""
    import csv
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'label_name', 'caption', 'image_path'])
        writer.writeheader()
        writer.writerows(captions)


def main():
    parser = argparse.ArgumentParser(description='Extract captions from syn_train_index JSON file')
    parser.add_argument('json_path', type=str, help='Path to the JSON file')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output file path (default: captions.json in same directory)')
    parser.add_argument('-f', '--format', type=str, choices=['txt', 'json', 'csv'], default='json',
                       help='Output format: txt (one caption per line), json, or csv (default: json)')
    parser.add_argument('--stats', action='store_true',
                       help='Print statistics about the captions')
    
    args = parser.parse_args()
    
    # Extract captions
    print(f"Loading JSON file: {args.json_path}")
    captions = extract_captions(args.json_path)
    print(f"Extracted {len(captions)} captions")
    
    # Determine output path
    if args.output is None:
        json_path = Path(args.json_path)
        output_path = json_path.parent / f"captions.{args.format}"
    else:
        output_path = Path(args.output)
    
    # Save captions
    output_path_str = str(output_path)
    print(f"Saving to: {output_path_str}")
    if args.format == 'txt':
        save_to_text(captions, output_path_str)
    elif args.format == 'json':
        save_to_json(captions, output_path_str)
    elif args.format == 'csv':
        save_to_csv(captions, output_path_str)
    
    print(f"Successfully saved {len(captions)} captions to {output_path_str}")
    
    # Print statistics if requested
    if args.stats:
        print("\nStatistics:")
        print(f"  Total captions: {len(captions)}")
        caption_lengths = [len(c['caption']) for c in captions]
        print(f"  Average caption length: {sum(caption_lengths) / len(caption_lengths):.1f} characters")
        print(f"  Min caption length: {min(caption_lengths)} characters")
        print(f"  Max caption length: {max(caption_lengths)} characters")
        
        # Count by label
        label_counts = {}
        for item in captions:
            label = item['label_name']
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"  Unique labels: {len(label_counts)}")
        if len(label_counts) <= 10:
            print("  Label distribution:")
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    {label}: {count}")


if __name__ == '__main__':
    main()

