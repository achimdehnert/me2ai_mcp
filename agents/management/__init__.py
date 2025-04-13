"""Management agents package for coordinating and optimizing other agents."""

from .routing.router import RouterAgent
from .teams.coordinator import TeamCoordinator
from .optimization.ml_optimizer import MLOptimizer
from .optimization.ab_testing import MultiVariantTest
from .optimization.multi_agent import MultiAgentOptimizer
from .optimization.feedback import FeedbackAnalyzer

__all__ = [
    'RouterAgent',
    'TeamCoordinator',
    'MLOptimizer',
    'MultiVariantTest',
    'MultiAgentOptimizer',
    'FeedbackAnalyzer'
]
