"""OpenAI LLM provider."""
import os
from typing import List, Dict, Any
from openai import OpenAI
from .base import LLMProvider

class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self):
        """Initialize OpenAI client."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4"
        
    async def generate_response(self, messages: List[Dict[str, Any]]) -> str:
        """Generate a response using OpenAI's API.
        
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
            raise Exception(f"Error generating response from OpenAI: {str(e)}")
