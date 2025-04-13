"""Base class for optimizers with feature toggling."""

from abc import ABC, abstractmethod
from typing import Optional
from .optimization_config import OptimizationConfig

class BaseOptimizer(ABC):
    """Base class for all optimizer components."""
    
    def __init__(self, feature_name: str):
        self.config = OptimizationConfig()
        self.feature_name = feature_name
        self._enabled = self.config.is_enabled(feature_name)
        
    @property
    def enabled(self) -> bool:
        """Check if this optimizer is enabled."""
        return self._enabled
    
    def enable(self) -> None:
        """Enable this optimizer."""
        self._enabled = True
        self.config.enable_feature(self.feature_name)
        self.config.save_config()
        
    def disable(self) -> None:
        """Disable this optimizer."""
        self._enabled = False
        self.config.disable_feature(self.feature_name)
        self.config.save_config()
        
    def check_enabled(self):
        """Decorator method to check if optimizer is enabled."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return None
                return func(*args, **kwargs)
            return wrapper
        return decorator
