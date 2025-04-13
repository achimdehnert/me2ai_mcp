"""Tools specifically designed for team coordination and collaboration."""
from typing import List, Dict, Any
from pydantic import BaseModel
from datetime import datetime
from .base import BaseTool

class TeamMetrics(BaseModel):
    """Metrics for team performance."""
    response_times: List[float]
    agreement_scores: List[float]
    contribution_counts: Dict[str, int]
    feedback_scores: List[float]

class TeamPerformanceMonitor(BaseTool):
    """Tool for tracking and analyzing team performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, TeamMetrics] = {}
        
    def execute(self, team_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Track and analyze team performance metrics.
        
        Args:
            team_name: Name of the team
            metrics: Dictionary containing metrics to track
            
        Returns:
            Dict containing analysis of team performance
        """
        if team_name not in self.metrics:
            self.metrics[team_name] = TeamMetrics(
                response_times=[],
                agreement_scores=[],
                contribution_counts={},
                feedback_scores=[]
            )
            
        team_metrics = self.metrics[team_name]
        
        # Update metrics
        if 'response_time' in metrics:
            team_metrics.response_times.append(metrics['response_time'])
        if 'agreement_score' in metrics:
            team_metrics.agreement_scores.append(metrics['agreement_score'])
        if 'contributor' in metrics:
            contributor = metrics['contributor']
            team_metrics.contribution_counts[contributor] = (
                team_metrics.contribution_counts.get(contributor, 0) + 1
            )
        if 'feedback_score' in metrics:
            team_metrics.feedback_scores.append(metrics['feedback_score'])
            
        # Analyze performance
        analysis = {
            'avg_response_time': sum(team_metrics.response_times) / len(team_metrics.response_times) if team_metrics.response_times else 0,
            'avg_agreement': sum(team_metrics.agreement_scores) / len(team_metrics.agreement_scores) if team_metrics.agreement_scores else 0,
            'contribution_distribution': team_metrics.contribution_counts,
            'avg_feedback': sum(team_metrics.feedback_scores) / len(team_metrics.feedback_scores) if team_metrics.feedback_scores else 0
        }
        
        return analysis

class ConsensusBuilder(BaseTool):
    """Tool for building consensus among team members."""
    
    def execute(self, perspectives: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze perspectives and identify areas of agreement/disagreement.
        
        Args:
            perspectives: List of dictionaries containing member perspectives
            
        Returns:
            Dict containing consensus analysis
        """
        # Extract key points from each perspective
        all_points = []
        for perspective in perspectives:
            points = perspective['content'].split('. ')
            all_points.extend((point, perspective['member']) for point in points)
            
        # Find common themes and disagreements
        themes = self._identify_themes(all_points)
        agreements = self._find_agreements(themes)
        disagreements = self._find_disagreements(themes)
        
        return {
            'agreements': agreements,
            'disagreements': disagreements,
            'agreement_score': len(agreements) / (len(agreements) + len(disagreements)) if agreements or disagreements else 0
        }
    
    def _identify_themes(self, points: List[tuple]) -> Dict[str, List[str]]:
        """Group points by common themes."""
        # In a real implementation, this would use NLP for theme identification
        themes = {}
        for point, member in points:
            theme = point.split()[0].lower()  # Simple theme extraction
            if theme not in themes:
                themes[theme] = []
            themes[theme].append((point, member))
        return themes
    
    def _find_agreements(self, themes: Dict[str, List[tuple]]) -> List[str]:
        """Identify points of agreement."""
        agreements = []
        for theme, points in themes.items():
            if len(set(member for _, member in points)) > 1:
                agreements.append(f"Agreement on {theme}: {points[0][0]}")
        return agreements
    
    def _find_disagreements(self, themes: Dict[str, List[tuple]]) -> List[str]:
        """Identify points of disagreement."""
        disagreements = []
        for theme, points in themes.items():
            if len(points) > 1 and len(set(p[0] for p in points)) > 1:
                disagreements.append(f"Disagreement on {theme}: {[p[0] for p in points]}")
        return disagreements
