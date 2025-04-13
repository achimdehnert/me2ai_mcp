"""Groq LLM provider."""
import os
from typing import List, Dict, Any
from groq import Groq
from .base import LLMProvider

class GroqProvider(LLMProvider):
    """Groq API provider."""
    
    def __init__(self):
        """Initialize Groq client."""
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "mixtral-8x7b-32768"
        
    async def generate_response(self, messages: List[Dict[str, Any]]) -> str:
        """Generate a response using Groq's API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            str: Generated response
            
        Raises:
            Exception: If there is an error generating the response
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating response from Groq: {str(e)}")
