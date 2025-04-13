"""Machine Learning based Agent Optimizer."""

from typing import Dict, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

@dataclass
class ConfigurationScore:
    """Score for a particular agent configuration."""
    temperature: float
    max_tokens: int
    presence_penalty: float
    frequency_penalty: float
    response_time: float
    success_rate: float
    user_satisfaction: float
    
class MLOptimizer:
    """Uses machine learning to predict optimal agent configurations."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100)
        self.training_data: List[ConfigurationScore] = []
        
    def add_training_data(self, score: ConfigurationScore) -> None:
        """Add a new configuration score to training data."""
        self.training_data.append(score)
        
    def train_model(self) -> None:
        """Train the model on collected configuration scores."""
        if len(self.training_data) < 10:
            raise ValueError("Need at least 10 data points to train model")
            
        # Prepare features (configurations) and targets (performance metrics)
        X = [[s.temperature, s.max_tokens, s.presence_penalty, s.frequency_penalty] 
             for s in self.training_data]
        y = [[s.response_time, s.success_rate, s.user_satisfaction] 
             for s in self.training_data]
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
    def predict_optimal_config(self, 
                             constraints: Optional[Dict] = None) -> Dict:
        """Predict optimal configuration based on learned patterns."""
        if not self.model:
            raise ValueError("Model not trained yet")
            
        # Generate candidate configurations
        candidates = self._generate_candidates(constraints)
        
        # Scale candidates
        candidates_scaled = self.scaler.transform(candidates)
        
        # Predict performance for each candidate
        predictions = self.model.predict(candidates_scaled)
        
        # Score predictions (weighted sum of metrics)
        scores = []
        for pred in predictions:
            response_time, success_rate, satisfaction = pred
            # Lower response time is better, higher others are better
            score = (-0.3 * response_time + 
                    0.4 * success_rate + 
                    0.3 * satisfaction)
            scores.append(score)
            
        # Get best configuration
        best_idx = np.argmax(scores)
        best_candidate = candidates[best_idx]
        
        return {
            "temperature": best_candidate[0],
            "max_tokens": int(best_candidate[1]),
            "presence_penalty": best_candidate[2],
            "frequency_penalty": best_candidate[3]
        }
        
    def _generate_candidates(self, 
                           constraints: Optional[Dict] = None) -> np.ndarray:
        """Generate candidate configurations to evaluate."""
        # Default ranges for parameters
        ranges = {
            "temperature": (0.1, 1.0, 10),
            "max_tokens": (50, 500, 10),
            "presence_penalty": (-2.0, 2.0, 10),
            "frequency_penalty": (-2.0, 2.0, 10)
        }
        
        # Apply constraints if provided
        if constraints:
            for param, (min_val, max_val) in constraints.items():
                if param in ranges:
                    ranges[param] = (min_val, max_val, 10)
                    
        # Generate grid of candidates
        temp_range = np.linspace(*ranges["temperature"])
        tokens_range = np.linspace(*ranges["max_tokens"])
        presence_range = np.linspace(*ranges["presence_penalty"])
        freq_range = np.linspace(*ranges["frequency_penalty"])
        
        # Create all combinations
        candidates = []
        for t in temp_range:
            for m in tokens_range:
                for p in presence_range:
                    for f in freq_range:
                        candidates.append([t, m, p, f])
                        
        return np.array(candidates)
