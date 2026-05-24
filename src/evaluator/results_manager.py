"""
Results management for evaluation experiments.

Provides utilities for:
- Formatted console output of results
- JSON serialization of experiment results
- Comparison tables between configurations
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentResult:
    """Container for a single experiment result."""
    name: str
    accuracy: float
    num_train_samples: int
    num_test_samples: int
    per_class_accuracy: Optional[Dict[int, Dict[str, Any]]] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class EvaluationResultsManager:
    """
    Manager for evaluation experiment results.
    
    Handles:
    - Collecting results from multiple experiments
    - Printing formatted summaries
    - Saving results to JSON files
    """
    
    def __init__(self, results_dir: str = 'results'):
        """
        Initialize results manager.
        
        Args:
            results_dir: Directory to save result files
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments: List[ExperimentResult] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_result(
        self,
        name: str,
        accuracy: float,
        num_train_samples: int,
        num_test_samples: int,
        per_class_accuracy: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> None:
        """
        Add an experiment result.
        
        Args:
            name: Experiment name (e.g., "Synthetic Only", "Zero-Shot")
            accuracy: Classification accuracy (0-1)
            num_train_samples: Number of training samples
            num_test_samples: Number of test samples
            per_class_accuracy: Dictionary mapping class_id to dict with 'accuracy' and 'name' keys
        """
        result = ExperimentResult(
            name=name,
            accuracy=accuracy,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            per_class_accuracy=per_class_accuracy
        )
        self.experiments.append(result)
    
    def set_metadata(self, **kwargs) -> None:
        """Set metadata for the experiment run."""
        self.metadata.update(kwargs)
    
    def print_summary(self) -> None:
        """Print formatted summary of all results."""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 80)
        
        # Print metadata
        if self.metadata:
            print("\n[Configuration]")
            for key, value in self.metadata.items():
                print(f"  {key}: {value}")
        
        # Print results table
        print("\n[Results]")
        print(f"{'Experiment':<40} {'Train Samples':<15} {'Accuracy':<15}")
        print("-" * 70)
        
        for result in self.experiments:
            train_str = str(result.num_train_samples) if result.num_train_samples > 0 else "N/A"
            acc_str = f"{result.accuracy:.4f} ({result.accuracy*100:.2f}%)"
            print(f"{result.name:<40} {train_str:<15} {acc_str:<15}")
        
        print("=" * 80)
        
        # Print key insights
        self._print_insights()
    
    def _print_insights(self) -> None:
        """Print key insights from the results."""
        if len(self.experiments) < 2:
            return
        
        print("\n[Key Insights]")
        
        # Find best and worst
        sorted_results = sorted(self.experiments, key=lambda x: x.accuracy, reverse=True)
        best = sorted_results[0]
        worst = sorted_results[-1]
        
        print(f"  Best:  {best.name} ({best.accuracy*100:.2f}%)")
        print(f"  Worst: {worst.name} ({worst.accuracy*100:.2f}%)")
        
        # Compare synthetic vs real if available
        syn_only = next((r for r in self.experiments if 'synthetic' in r.name.lower() and 'real' not in r.name.lower()), None)
        real_only = next((r for r in self.experiments if 'real' in r.name.lower() and 'synthetic' not in r.name.lower()), None)
        
        if syn_only and real_only:
            diff = syn_only.accuracy - real_only.accuracy
            comparison = "higher" if diff > 0 else "lower"
            print(f"  Synthetic vs Real: {abs(diff)*100:.2f}% {comparison}")
    
    def save_results(self, filename_prefix: str = "eval_results") -> str:
        """
        Save results to JSON file.
        
        Args:
            filename_prefix: Prefix for the output filename
        
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        output = {
            'timestamp': timestamp,
            'metadata': self.metadata,
            'results': [r.to_dict() for r in self.experiments],
            'summary': {
                'num_experiments': len(self.experiments),
                'best_accuracy': max(r.accuracy for r in self.experiments) if self.experiments else 0,
                'worst_accuracy': min(r.accuracy for r in self.experiments) if self.experiments else 0,
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {filepath}")
        return str(filepath)
    
    def print_comparison_table(
        self,
        baseline_name: str,
        compare_names: List[str]
    ) -> None:
        """
        Print comparison table against a baseline.
        
        Args:
            baseline_name: Name of baseline experiment
            compare_names: Names of experiments to compare
        """
        baseline = next((r for r in self.experiments if r.name == baseline_name), None)
        if not baseline:
            print(f"Baseline '{baseline_name}' not found")
            return
        
        print(f"\n[Comparison vs {baseline_name}]")
        print(f"{'Experiment':<35} {'Accuracy':<15} {'vs Baseline':<15}")
        print("-" * 65)
        
        print(f"{baseline.name:<35} {baseline.accuracy*100:.2f}%         baseline")
        
        for name in compare_names:
            result = next((r for r in self.experiments if r.name == name), None)
            if result:
                diff = result.accuracy - baseline.accuracy
                diff_str = f"{diff*100:+.2f}%"
                print(f"{result.name:<35} {result.accuracy*100:.2f}%         {diff_str}")

