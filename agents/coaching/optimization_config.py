"""Configuration system for agent optimization features."""

from typing import Dict, Optional
from dataclasses import dataclass
import os
import json

@dataclass
class OptimizationFeatures:
    """Configuration for optimization features."""
    ml_optimization: bool = False
    ab_testing: bool = False
    multi_agent_optimization: bool = False
    feedback_analysis: bool = False
    data_collection: bool = True  # Always collect data by default
    
    # Feature-specific settings
    ml_training_frequency: int = 24  # hours
    ab_test_sample_size: int = 1000
    team_analysis_interval: int = 6  # hours
    feedback_batch_size: int = 50

class OptimizationConfig:
    """Controls optimization features across different environments."""
    
    _instance = None  # Singleton instance
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.environment = os.getenv('ME2AI_ENV', 'development')
            self.features = self._load_config()
            
    def _load_config(self) -> OptimizationFeatures:
        """Load configuration based on environment."""
        config_path = os.getenv(
            'ME2AI_CONFIG_PATH',
            'config/optimization.json'
        )
        
        # Default configurations for different environments
        default_configs = {
            'development': OptimizationFeatures(
                ml_optimization=True,
                ab_testing=True,
                multi_agent_optimization=True,
                feedback_analysis=True,
                data_collection=True
            ),
            'staging': OptimizationFeatures(
                ml_optimization=False,
                ab_testing=True,
                multi_agent_optimization=True,
                feedback_analysis=True,
                data_collection=True
            ),
            'production': OptimizationFeatures(
                ml_optimization=False,
                ab_testing=False,
                multi_agent_optimization=False,
                feedback_analysis=False,
                data_collection=True  # Still collect data in production
            )
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                return OptimizationFeatures(**config_data.get(
                    self.environment,
                    default_configs[self.environment].__dict__
                ))
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            
        return default_configs.get(
            self.environment,
            default_configs['production']
        )
    
    def is_enabled(self, feature: str) -> bool:
        """Check if a specific optimization feature is enabled."""
        return getattr(self.features, feature, False)
    
    def enable_feature(self, feature: str) -> None:
        """Enable a specific optimization feature."""
        if hasattr(self.features, feature):
            setattr(self.features, feature, True)
            
    def disable_feature(self, feature: str) -> None:
        """Disable a specific optimization feature."""
        if hasattr(self.features, feature):
            setattr(self.features, feature, False)
            
    def update_settings(self, settings: Dict) -> None:
        """Update feature-specific settings."""
        for key, value in settings.items():
            if hasattr(self.features, key):
                setattr(self.features, key, value)
                
    def save_config(self) -> None:
        """Save current configuration to file."""
        config_path = os.getenv(
            'ME2AI_CONFIG_PATH',
            'config/optimization.json'
        )
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Load existing config if it exists
        existing_config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                existing_config = json.load(f)
                
        # Update config for current environment
        existing_config[self.environment] = self.features.__dict__
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(existing_config, f, indent=2)
            
    @property
    def settings(self) -> Dict:
        """Get current feature settings."""
        return self.features.__dict__
