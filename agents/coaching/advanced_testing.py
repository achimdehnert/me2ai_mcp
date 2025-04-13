"""Advanced A/B testing strategies for agent optimization."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
from dataclasses import dataclass
import time

@dataclass
class TestResult:
    """Results from a single test iteration."""
    variant: str
    metrics: Dict[str, float]
    timestamp: float
    user_id: str
    
class MultiVariantTest:
    """Supports advanced A/B/n testing with multiple variants."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.variants: Dict[str, List[TestResult]] = {}
        self.confidence_level = confidence_level
        self.start_time = time.time()
        
    def add_variant(self, variant_name: str) -> None:
        """Add a new variant to the test."""
        if variant_name not in self.variants:
            self.variants[variant_name] = []
            
    def record_result(self, result: TestResult) -> None:
        """Record a test result for a variant."""
        if result.variant not in self.variants:
            raise ValueError(f"Unknown variant: {result.variant}")
        self.variants[result.variant].append(result)
        
    def get_results(self) -> Dict:
        """Get statistical analysis of test results."""
        results = {
            "sample_sizes": {},
            "metrics": {},
            "significance": {},
            "recommendations": []
        }
        
        # Calculate sample sizes
        for variant, data in self.variants.items():
            results["sample_sizes"][variant] = len(data)
            
        # Calculate metrics for each variant
        for metric in ["response_time", "success_rate", "satisfaction"]:
            metric_data = {}
            for variant, data in self.variants.items():
                values = [r.metrics[metric] for r in data]
                metric_data[variant] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "ci": stats.t.interval(
                        self.confidence_level, 
                        len(values)-1,
                        loc=np.mean(values),
                        scale=stats.sem(values)
                    ) if len(values) > 1 else None
                }
            results["metrics"][metric] = metric_data
            
        # Perform statistical tests
        control = list(self.variants.keys())[0]  # First variant is control
        for variant in list(self.variants.keys())[1:]:
            for metric in ["response_time", "success_rate", "satisfaction"]:
                control_data = [r.metrics[metric] for r in self.variants[control]]
                variant_data = [r.metrics[metric] for r in self.variants[variant]]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(control_data, variant_data)
                
                results["significance"][f"{variant}_vs_control_{metric}"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < (1 - self.confidence_level)
                }
                
        # Generate recommendations
        self._add_recommendations(results)
        
        return results
    
    def _add_recommendations(self, results: Dict) -> None:
        """Add recommendations based on test results."""
        control = list(self.variants.keys())[0]
        
        for variant in list(self.variants.keys())[1:]:
            improvements = []
            
            for metric in ["response_time", "success_rate", "satisfaction"]:
                control_mean = results["metrics"][metric][control]["mean"]
                variant_mean = results["metrics"][metric][variant]["mean"]
                sig_key = f"{variant}_vs_control_{metric}"
                
                if results["significance"][sig_key]["significant"]:
                    diff_pct = ((variant_mean - control_mean) / control_mean) * 100
                    if (metric == "response_time" and diff_pct < 0) or \
                       (metric != "response_time" and diff_pct > 0):
                        improvements.append(
                            f"{metric}: {abs(diff_pct):.1f}% {'decrease' if metric == 'response_time' else 'increase'}"
                        )
                        
            if improvements:
                results["recommendations"].append({
                    "variant": variant,
                    "improvements": improvements,
                    "confidence": f"{self.confidence_level * 100}%"
                })
            
    def should_continue(self, 
                       min_samples: int = 100,
                       max_duration: float = 86400) -> bool:
        """Determine if testing should continue."""
        # Check if we have minimum sample size
        for results in self.variants.values():
            if len(results) < min_samples:
                return True
                
        # Check if we've exceeded max duration
        if time.time() - self.start_time > max_duration:
            return False
            
        # Check if we have statistical significance
        results = self.get_results()
        for sig_test in results["significance"].values():
            if not sig_test["significant"]:
                return True
                
        return False
