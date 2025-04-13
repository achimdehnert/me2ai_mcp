"""Base agent interface."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.memory import BaseMemory
from me2ai.llms.base import LLMProvider
from me2ai.tools import Tool

class BaseAgent(ABC):
    """Abstract base class for agents."""
    
    def __init__(
        self,
        role: str,
        system_prompt: str,
        llm_provider: LLMProvider,
        memory: Optional[BaseMemory] = None,
        tools: Optional[List[Tool]] = None
    ):
        """Initialize the base agent.
        
        Args:
            role: The role of the agent (e.g., "Moderator", "Life Coach")
            system_prompt: The system prompt that defines the agent's behavior
            llm_provider: The LLM provider to use for generating responses
            memory: Optional memory instance for maintaining conversation history
            tools: Optional list of tools available to the agent
        """
        self.role = role
        self.base_prompt = system_prompt
        self.llm_provider = llm_provider
        self.memory = memory
        self.tools = tools or []
        
        # Add tool descriptions to system prompt if tools are provided
        if tools:
            tool_descriptions = "\n\nAvailable tools:\n" + "\n".join(
                f"- {tool.name}: {tool.description}" for tool in tools
            )
            self.system_prompt = system_prompt + tool_descriptions
        else:
            self.system_prompt = system_prompt
    
    def use_tool(self, tool_name: str, **kwargs: Any) -> str:
        """Use a specific tool.
        
        Args:
            tool_name: Name of the tool to use
            **kwargs: Arguments to pass to the tool
            
        Returns:
            str: Result from the tool
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.run(**kwargs)
        return f"Tool '{tool_name}' not found."
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names.
        
        Returns:
            List[str]: Names of available tools
        """
        return [tool.name for tool in self.tools]
    
    async def _build_messages(self, user_input: str) -> List[Dict[str, Any]]:
        """Build message list for LLM.
        
        Args:
            user_input: User's input message
            
        Returns:
            List of message dictionaries
        """
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history if available
        if self.memory:
            history = self.memory.get_messages()
            messages.extend(history)
            
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
        
    async def _generate_response(self, prompt: str) -> str:
        """Generate response using LLM.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response
        """
        messages = await self._build_messages(prompt)
        response = await self.llm_provider.generate(messages)
        
        if self.memory:
            self.memory.add_message("user", prompt)
            self.memory.add_message("assistant", response)
            
        return response
        
    @abstractmethod
    async def respond(self, message: str, context: Optional[str] = None) -> str:
        """Generate a response to the given message.
        
        Args:
            message: The input message to respond to
            context: Optional additional context
            
        Returns:
            str: The generated response
        """
        pass
