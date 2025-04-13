"""Base tool interface."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseTool(ABC):
    """Base class for all tools."""

    def __init__(self, name: str = "", description: str = "", parameters: Optional[Dict[str, Any]] = None):
        """Initialize tool.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: Tool parameters schema
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Run the tool with given arguments."""
        pass

    def validate_input(self, **kwargs: Any) -> bool:
        """Validate input arguments."""
        return True

    def format_output(self, result: Any) -> Any:
        """Format tool output."""
        return result
