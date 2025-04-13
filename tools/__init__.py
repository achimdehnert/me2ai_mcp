"""Tools package for expert agents.

This package contains various tools that agents can use to enhance their capabilities,
such as web search, language translation, SEO analysis, etc.
"""

from typing import Protocol, List, Dict, Any

class Tool(Protocol):
    """Protocol defining the interface for tools."""
    
    name: str
    description: str
    
    def run(self, **kwargs: Any) -> str:
        """Run the tool with the given arguments.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            str: Result of running the tool
        """
        ...
