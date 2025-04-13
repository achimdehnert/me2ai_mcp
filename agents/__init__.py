"""Agent module initialization."""
from .base import BaseAgent
from .individual import (
    GermanProfessor,
    DatingExpert,
    SEOExpert,
    Researcher,
    ContentStrategistAgent,
    DataAnalystAgent
)
from .management.teams.business.general import BusinessTeam
from .management.teams.language.general import LanguageTeam
from .management.teams.relationship.general import RelationshipTeam
from .management.teams.seo.general import SEOTeam
from .management.routing.router import RouterAgent
from .factory import AgentFactory

__all__ = [
    'BaseAgent',
    'GermanProfessor',
    'DatingExpert',
    'SEOExpert',
    'Researcher',
    'ContentStrategistAgent',
    'DataAnalystAgent',
    'BusinessTeam',
    'LanguageTeam',
    'RelationshipTeam',
    'SEOTeam',
    'RouterAgent',
    'AgentFactory'
]
