"""German language and culture expert agent."""
from typing import Optional

from me2ai.llms.base import LLMProvider
from me2ai.memory import ConversationMemory
from me2ai.agents.base import BaseAgent
from me2ai.tools.language_tools import GermanDictionaryTool, GrammarCheckerTool
from me2ai.tools.web_tools import TranslationTool, WebSearchTool

class GermanProfessor(BaseAgent):
    """German language and culture expert."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        memory: Optional[ConversationMemory] = None
    ):
        """Initialize the agent.
        
        Args:
            llm_provider: LLM provider to use
            memory: Optional conversation memory
        """
        system_prompt = """You are a German language professor and cultural expert. 
        You help students learn German language and understand German culture.
        
        When teaching language:
        - Focus on practical usage and real-world examples
        - Explain grammar concepts clearly and simply
        - Provide pronunciation tips using IPA notation
        - Encourage active practice and immersion
        
        When discussing culture:
        - Share insights about German customs and traditions
        - Explain historical context when relevant
        - Highlight regional differences within German-speaking countries
        - Recommend authentic cultural resources
        
        Always maintain a patient, encouraging, and academic tone.
        """
        
        tools = [
            GermanDictionaryTool(),
            GrammarCheckerTool(),
            TranslationTool(),
            WebSearchTool()
        ]
        
        super().__init__(
            role="German Professor",
            system_prompt=system_prompt,
            llm_provider=llm_provider,
            memory=memory,
            tools=tools
        )

    async def respond(self, message: str, context: Optional[str] = None) -> str:
        """Generate a response to the user's message.
        
        Args:
            message: The user's message
            context: Optional additional context
            
        Returns:
            str: The agent's response
        """
        return await self._generate_response(message)
