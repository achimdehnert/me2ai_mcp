"""Base class for optimization components."""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class OptimizationMetrics:
    """Metrics for measuring optimization performance."""
    response_time: float  # Average response time in seconds
    success_rate: float  # Percentage of successful optimizations
    error_rate: float  # Percentage of errors encountered
    improvement_score: float  # Quantified improvement score (0-100)

class BaseOptimizer(ABC):
    """Base class for all optimizer components."""
    
    def __init__(self):
        """Initialize the base optimizer."""
        self.metrics: Dict[str, OptimizationMetrics] = {}
        self.optimization_history: List[Dict] = []
    
    @abstractmethod
    def optimize(self, target_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize the target component with the given configuration.
        
        Args:
            target_id: ID of the component to optimize
            config: Configuration parameters for optimization
            
        Returns:
            Dict containing optimization results
        """
        pass
    
    @abstractmethod
    def analyze_performance(self, target_id: str) -> Dict[str, Any]:
        """Analyze performance of the target component.
        
        Args:
            target_id: ID of the component to analyze
            
        Returns:
            Dict containing performance analysis results
        """
        pass
    
    def record_metrics(self, target_id: str, metrics: OptimizationMetrics):
        """Record performance metrics for a target component.
        
        Args:
            target_id: ID of the component
            metrics: Performance metrics to record
        """
        self.metrics[target_id] = metrics
    
    def get_metrics(self, target_id: str) -> Optional[OptimizationMetrics]:
        """Get recorded metrics for a target component.
        
        Args:
            target_id: ID of the component
            
        Returns:
            Recorded metrics if available, None otherwise
        """
        return self.metrics.get(target_id)
    
    def get_optimization_history(self, target_id: Optional[str] = None) -> List[Dict]:
        """Get optimization history for a specific or all components.
        
        Args:
            target_id: Optional ID to filter history for a specific component
            
        Returns:
            List of optimization history entries
        """
        if target_id:
            return [entry for entry in self.optimization_history if entry["target_id"] == target_id]
        return self.optimization_history
