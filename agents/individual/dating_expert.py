"""Dating and relationship expert agent."""
from typing import Optional

from me2ai.llms.base import LLMProvider
from me2ai.memory import ConversationMemory
from me2ai.agents.base import BaseAgent
from me2ai.tools.dating_tools import ProfileAnalysisTool, ConversationCoachTool

class DatingExpert(BaseAgent):
    """Dating and relationship expert agent."""

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
        system_prompt = """You are a dating and relationship expert.
        You help people navigate dating, relationships, and personal growth.
        
        When giving advice:
        - Focus on healthy relationship dynamics
        - Promote clear communication and boundaries
        - Consider emotional and psychological factors
        - Encourage self-reflection and growth
        
        When discussing specific situations:
        - Ask clarifying questions when needed
        - Provide actionable suggestions
        - Consider cultural and personal context
        - Maintain empathy and sensitivity
        
        Always maintain a supportive and non-judgmental tone.
        """
        
        tools = [
            ProfileAnalysisTool(),
            ConversationCoachTool()
        ]
        
        super().__init__(
            role="Dating Expert",
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
