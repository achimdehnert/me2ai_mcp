"""Factory functions for creating expert agents."""
from typing import Optional, Dict, Any

from .base import BaseAgent
from .individual.seo_expert import SEOExpert
from .individual.dating_expert import DatingExpert
from .individual.german_professor import GermanProfessor
from .individual.content_strategist import ContentStrategistAgent
from .individual.data_analyst import DataAnalystAgent
from .individual.researcher import Researcher
from .individual.web_scraper import WebScraperAgent

def create_expert_agent(agent_type: str, config: Optional[Dict[str, Any]] = None) -> BaseAgent:
    """Create an expert agent of the specified type.
    
    Args:
        agent_type: Type of expert agent to create
        config: Optional configuration parameters
        
    Returns:
        BaseAgent: Instantiated expert agent
    
    Raises:
        ValueError: If agent_type is not recognized
    """
    config = config or {}
    
    agent_map = {
        "seo": SEOExpert,
        "dating": DatingExpert,
        "german": GermanProfessor,
        "content": ContentStrategistAgent,
        "data": DataAnalystAgent,
        "research": Researcher,
        "scraper": WebScraperAgent
    }
    
    if agent_type not in agent_map:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    agent_class = agent_map[agent_type]
    return agent_class(**config)
