"""A/B testing and multi-variant testing functionality."""
from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass
from .base import BaseOptimizer, OptimizationMetrics

@dataclass
class TestVariant:
    """A variant in an A/B or multi-variant test."""
    name: str
    config: Dict[str, Any]
    metrics: Optional[OptimizationMetrics] = None

class MultiVariantTest:
    """Handles multi-variant testing of different configurations."""
    
    def __init__(self, name: str, variants: List[TestVariant]):
        """Initialize a multi-variant test.
        
        Args:
            name: Name of the test
            variants: List of test variants
        """
        self.name = name
        self.variants = variants
        self.start_time = None
        self.end_time = None
        self.results = {}
    
    def start(self):
        """Start the test."""
        self.start_time = time.time()
    
    def stop(self):
        """Stop the test."""
        self.end_time = time.time()
    
    def record_metrics(self, variant_name: str, metrics: OptimizationMetrics):
        """Record metrics for a specific variant.
        
        Args:
            variant_name: Name of the variant
            metrics: Performance metrics to record
        """
        for variant in self.variants:
            if variant.name == variant_name:
                variant.metrics = metrics
                break
    
    def get_results(self) -> Dict[str, Any]:
        """Get the test results.
        
        Returns:
            Dict containing test results and analysis
        """
        if not self.start_time or not self.end_time:
            return {"error": "Test not completed"}
        
        results = {
            "name": self.name,
            "duration": self.end_time - self.start_time,
            "variants": {}
        }
        
        # Find the best performing variant
        best_variant = None
        best_score = -1
        
        for variant in self.variants:
            if variant.metrics:
                # Calculate a simple performance score
                score = (
                    variant.metrics.success_rate * 0.4 +
                    (1 - variant.metrics.error_rate) * 0.3 +
                    (1 / variant.metrics.response_time) * 0.2 +
                    (variant.metrics.improvement_score / 100) * 0.1
                )
                
                results["variants"][variant.name] = {
                    "metrics": {
                        "response_time": variant.metrics.response_time,
                        "success_rate": variant.metrics.success_rate,
                        "error_rate": variant.metrics.error_rate,
                        "improvement_score": variant.metrics.improvement_score
                    },
                    "score": score
                }
                
                if score > best_score:
                    best_score = score
                    best_variant = variant.name
        
        if best_variant:
            results["best_variant"] = best_variant
            results["improvement"] = self._calculate_improvement(best_variant)
        
        return results
    
    def _calculate_improvement(self, best_variant: str) -> float:
        """Calculate improvement percentage over baseline.
        
        Args:
            best_variant: Name of the best performing variant
            
        Returns:
            float: Improvement percentage
        """
        baseline = None
        best = None
        
        for variant in self.variants:
            if variant.name == "baseline" and variant.metrics:
                baseline = variant.metrics
            elif variant.name == best_variant and variant.metrics:
                best = variant.metrics
        
        if baseline and best:
            # Calculate overall improvement percentage
            baseline_score = (
                baseline.success_rate * 0.4 +
                (1 - baseline.error_rate) * 0.3 +
                (1 / baseline.response_time) * 0.2 +
                (baseline.improvement_score / 100) * 0.1
            )
            
            best_score = (
                best.success_rate * 0.4 +
                (1 - best.error_rate) * 0.3 +
                (1 / best.response_time) * 0.2 +
                (best.improvement_score / 100) * 0.1
            )
            
            return ((best_score - baseline_score) / baseline_score) * 100
            
        return 0.0
