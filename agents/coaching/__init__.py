"""Coaching agents package."""

from .base import CoachingAgent
from .moderator import ModeratorAgent
from .life_coach import LifeCoachAgent
from .optimizer import AgentOptimizer

__all__ = [
    'CoachingAgent',
    'ModeratorAgent',
    'LifeCoachAgent',
    'AgentOptimizer'
]
