"""Feedback analysis and optimization system."""
from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass
from .base import BaseOptimizer, OptimizationMetrics

@dataclass
class FeedbackEntry:
    """A single piece of feedback."""
    timestamp: float
    source: str
    content: str
    sentiment: float  # -1.0 to 1.0
    category: str

class FeedbackAnalyzer(BaseOptimizer):
    """Analyzer for user and system feedback."""
    
    def __init__(self):
        """Initialize the feedback analyzer."""
        super().__init__()
        self.feedback_history: List[FeedbackEntry] = []
        self.sentiment_trends: Dict[str, List[float]] = {}
    
    def optimize(self, target_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize based on feedback analysis.
        
        Args:
            target_id: ID of the component to optimize
            config: Current configuration parameters
            
        Returns:
            Dict containing optimization results
        """
        timestamp = time.time()
        
        # Analyze feedback patterns
        feedback_analysis = self._analyze_feedback_patterns(target_id)
        
        # Generate optimized configuration
        optimized_config = self._optimize_from_feedback(config, feedback_analysis)
        
        results = {
            "target_id": target_id,
            "timestamp": timestamp,
            "feedback_analysis": feedback_analysis,
            "original_config": config,
            "optimized_config": optimized_config,
            "predicted_improvement": self._estimate_improvement(feedback_analysis)
        }
        
        self.optimization_history.append(results)
        return results
    
    def analyze_performance(self, target_id: str) -> Dict[str, Any]:
        """Analyze performance through feedback.
        
        Args:
            target_id: ID of the component to analyze
            
        Returns:
            Dict containing performance analysis
        """
        metrics = self.get_metrics(target_id)
        if not metrics:
            return {"error": "No metrics available for analysis"}
        
        analysis = {
            "metrics": {
                "response_time": metrics.response_time,
                "success_rate": metrics.success_rate,
                "error_rate": metrics.error_rate,
                "improvement_score": metrics.improvement_score
            },
            "feedback_summary": self._summarize_feedback(target_id),
            "sentiment_analysis": self._analyze_sentiment(target_id),
            "recommendations": self._generate_feedback_recommendations(metrics)
        }
        
        return analysis
    
    def add_feedback(self, feedback: FeedbackEntry):
        """Add a new feedback entry.
        
        Args:
            feedback: Feedback entry to add
        """
        self.feedback_history.append(feedback)
        
        # Update sentiment trends
        if feedback.source not in self.sentiment_trends:
            self.sentiment_trends[feedback.source] = []
        self.sentiment_trends[feedback.source].append(feedback.sentiment)
    
    def _analyze_feedback_patterns(self, target_id: str) -> Dict[str, Any]:
        """Analyze patterns in feedback data.
        
        Args:
            target_id: ID of the component
            
        Returns:
            Dict containing feedback pattern analysis
        """
        relevant_feedback = [
            f for f in self.feedback_history
            if f.source == target_id
        ]
        
        if not relevant_feedback:
            return {"error": "No feedback available"}
        
        patterns = {
            "total_feedback": len(relevant_feedback),
            "sentiment_distribution": self._calculate_sentiment_distribution(relevant_feedback),
            "common_topics": self._extract_common_topics(relevant_feedback),
            "trend": self._calculate_sentiment_trend(relevant_feedback)
        }
        
        return patterns
    
    def _optimize_from_feedback(self, config: Dict[str, Any], 
                              feedback_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized configuration based on feedback.
        
        Args:
            config: Current configuration
            feedback_analysis: Feedback pattern analysis
            
        Returns:
            Dict containing optimized configuration
        """
        optimized = config.copy()
        
        # Apply feedback-based optimizations
        if "sentiment_distribution" in feedback_analysis:
            sentiment = feedback_analysis["sentiment_distribution"]
            
            if sentiment.get("negative", 0) > 0.3:  # High negative feedback
                # Adjust configuration to address common complaints
                if "response_style" in optimized:
                    optimized["response_style"] = "more_detailed"
                if "error_tolerance" in optimized:
                    optimized["error_tolerance"] = "strict"
        
        return optimized
    
    def _estimate_improvement(self, feedback_analysis: Dict[str, Any]) -> float:
        """Estimate potential improvement from feedback-based optimization.
        
        Args:
            feedback_analysis: Feedback pattern analysis
            
        Returns:
            float: Estimated improvement percentage
        """
        if "sentiment_distribution" not in feedback_analysis:
            return 0.0
            
        sentiment = feedback_analysis["sentiment_distribution"]
        current_positive = sentiment.get("positive", 0)
        
        # Estimate potential improvement
        potential_improvement = (1 - current_positive) * 0.5  # Conservative estimate
        return potential_improvement * 100
    
    def _summarize_feedback(self, target_id: str) -> Dict[str, Any]:
        """Generate a summary of feedback.
        
        Args:
            target_id: ID of the component
            
        Returns:
            Dict containing feedback summary
        """
        relevant_feedback = [
            f for f in self.feedback_history
            if f.source == target_id
        ]
        
        return {
            "total_entries": len(relevant_feedback),
            "categories": self._summarize_categories(relevant_feedback),
            "sentiment": self._calculate_average_sentiment(relevant_feedback)
        }
    
    def _analyze_sentiment(self, target_id: str) -> Dict[str, Any]:
        """Analyze sentiment trends.
        
        Args:
            target_id: ID of the component
            
        Returns:
            Dict containing sentiment analysis
        """
        if target_id not in self.sentiment_trends:
            return {"error": "No sentiment data available"}
            
        sentiments = self.sentiment_trends[target_id]
        
        return {
            "current": sentiments[-1] if sentiments else 0,
            "trend": self._calculate_trend(sentiments),
            "volatility": self._calculate_volatility(sentiments)
        }
    
    def _generate_feedback_recommendations(self, metrics: OptimizationMetrics) -> List[Dict[str, str]]:
        """Generate recommendations based on feedback analysis.
        
        Args:
            metrics: Current performance metrics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if metrics.success_rate < 0.9:
            recommendations.append({
                "area": "user_satisfaction",
                "suggestion": "Implement more user-friendly error messages"
            })
            
        if metrics.response_time > 2.0:
            recommendations.append({
                "area": "performance",
                "suggestion": "Add progress indicators for long-running operations"
            })
            
        return recommendations
    
    def _calculate_sentiment_distribution(self, feedback: List[FeedbackEntry]) -> Dict[str, float]:
        """Calculate distribution of sentiment in feedback.
        
        Args:
            feedback: List of feedback entries
            
        Returns:
            Dict containing sentiment distribution
        """
        total = len(feedback)
        if total == 0:
            return {}
            
        distribution = {
            "positive": len([f for f in feedback if f.sentiment > 0.3]) / total,
            "neutral": len([f for f in feedback if -0.3 <= f.sentiment <= 0.3]) / total,
            "negative": len([f for f in feedback if f.sentiment < -0.3]) / total
        }
        
        return distribution
    
    def _extract_common_topics(self, feedback: List[FeedbackEntry]) -> List[Dict[str, Any]]:
        """Extract common topics from feedback.
        
        Args:
            feedback: List of feedback entries
            
        Returns:
            List of common topics with frequencies
        """
        topics = {}
        for entry in feedback:
            if entry.category not in topics:
                topics[entry.category] = 0
            topics[entry.category] += 1
        
        return [
            {"topic": topic, "frequency": count}
            for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True)
        ]
    
    def _calculate_sentiment_trend(self, feedback: List[FeedbackEntry]) -> str:
        """Calculate trend in sentiment over time.
        
        Args:
            feedback: List of feedback entries
            
        Returns:
            str: Trend description
        """
        if len(feedback) < 2:
            return "insufficient_data"
            
        # Sort by timestamp
        sorted_feedback = sorted(feedback, key=lambda x: x.timestamp)
        
        # Compare recent vs older sentiment
        midpoint = len(sorted_feedback) // 2
        old_sentiment = sum(f.sentiment for f in sorted_feedback[:midpoint]) / midpoint
        recent_sentiment = sum(f.sentiment for f in sorted_feedback[midpoint:]) / (len(sorted_feedback) - midpoint)
        
        if recent_sentiment > old_sentiment + 0.1:
            return "improving"
        elif recent_sentiment < old_sentiment - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _summarize_categories(self, feedback: List[FeedbackEntry]) -> Dict[str, int]:
        """Summarize feedback by category.
        
        Args:
            feedback: List of feedback entries
            
        Returns:
            Dict containing category counts
        """
        categories = {}
        for entry in feedback:
            if entry.category not in categories:
                categories[entry.category] = 0
            categories[entry.category] += 1
        
        return categories
    
    def _calculate_average_sentiment(self, feedback: List[FeedbackEntry]) -> float:
        """Calculate average sentiment score.
        
        Args:
            feedback: List of feedback entries
            
        Returns:
            float: Average sentiment score
        """
        if not feedback:
            return 0.0
        return sum(f.sentiment for f in feedback) / len(feedback)
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend in a series of values.
        
        Args:
            values: List of values
            
        Returns:
            str: Trend description
        """
        if len(values) < 2:
            return "insufficient_data"
            
        start = sum(values[:3]) / 3 if len(values) >= 3 else values[0]
        end = sum(values[-3:]) / 3 if len(values) >= 3 else values[-1]
        
        if end > start + 0.1:
            return "improving"
        elif end < start - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility in a series of values.
        
        Args:
            values: List of values
            
        Returns:
            float: Volatility score
        """
        if len(values) < 2:
            return 0.0
            
        differences = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
        return sum(differences) / len(differences)
