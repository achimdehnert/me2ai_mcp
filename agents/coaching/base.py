"""Base class for coaching agents."""
from typing import Optional, Dict, Any
from langchain_core.memory import BaseMemory
from langchain_core.messages import HumanMessage, AIMessage
from ..base import BaseAgent
from me2ai.llms.base import LLMProvider

class CoachingAgent(BaseAgent):
    """Base class for coaching agents with memory management."""
    
    def respond(self, message: str, context: Optional[str] = None) -> str:
        """Generate a response using the LLM provider.
        
        Args:
            message: The input message to respond to
            context: Optional additional context
            
        Returns:
            str: The generated response
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        # Add context from memory if available
        if self.memory and hasattr(self.memory, 'chat_history'):
            for msg in self.memory.chat_history.messages:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})
        
        # Add current context and message
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": message})
        
        # Generate response
        response = self.llm_provider.generate_response(messages)
        
        # Update memory if available
        if self.memory:
            self.memory.save_context(
                {"input": message},
                {"output": response}
            )
        
        return response
