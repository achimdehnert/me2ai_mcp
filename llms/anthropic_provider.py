"""Anthropic LLM provider implementation."""
import os
import anthropic
from typing import List, Dict, Any
from .base import LLMProvider

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self):
        """Initialize the Anthropic provider."""
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-3-opus-20240229"
        
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response using Anthropic's API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for Anthropic's API
            
        Returns:
            str: The generated response
        """
        try:
            # Convert messages to Anthropic format
            system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
            human_messages = [m for m in messages if m["role"] == "user"]
            assistant_messages = [m for m in messages if m["role"] == "assistant"]
            
            # Build the conversation
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            for h, a in zip(human_messages, assistant_messages + [None]):
                messages.append({"role": "user", "content": h["content"]})
                if a:
                    messages.append({"role": "assistant", "content": a["content"]})
            
            # Generate response
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1024)
            )
            return response.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"
