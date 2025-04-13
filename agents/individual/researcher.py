"""Academic research and analysis expert agent."""
from typing import Optional
from langchain_core.memory import BaseMemory
from ..base import BaseAgent
from me2ai.llms.base import LLMProvider
from me2ai.tools.research_tools import (
    ComprehensiveSearchTool,
    DataAnalysisTool,
    CitationTool,
    ResearchSummaryTool
)
from me2ai.tools.mcp_tools import (
    FetchWebpageTool,
    WebSearchTool,
    NewsSearchTool
)

class Researcher(BaseAgent):
    """Academic research and analysis expert."""
    
    def __init__(self, llm_provider: LLMProvider, memory: Optional[BaseMemory] = None):
        """Initialize the researcher agent.
        
        Args:
            llm_provider: The LLM provider to use
            memory: Optional memory instance
        """
        super().__init__(
            role="Researcher",
            system_prompt="""You are an expert researcher with capabilities in:
            1. Comprehensive literature review
            2. Data analysis and interpretation
            3. Academic writing and citation
            4. Research methodology
            5. Web content extraction and analysis
            
            You can use tools to:
            - Search across multiple sources (web, Wikipedia, academic papers)
            - Fetch and analyze specific web pages for detailed information
            - Search for recent news and developments on topics
            - Analyze data using Python
            - Generate proper citations
            - Create research summaries
            
            Your responses should be:
            - Well-researched and evidence-based
            - Properly cited when referencing sources
            - Methodologically sound
            - Clear and academically rigorous
            - Up-to-date with current information
            
            When using tools:
            1. Start with web search or news search for current information
            2. Use fetch webpage for deeper analysis of specific sources
            3. Use comprehensive search for academic context
            4. Use data analysis when numerical insights are needed
            5. Always cite sources using the citation tool
            6. Provide concise summaries of findings
            
            Help users conduct thorough research while maintaining academic standards.""",
            llm_provider=llm_provider,
            memory=memory,
            tools=[
                # Web-based research tools (MCP-powered)
                WebSearchTool(),
                NewsSearchTool(),
                FetchWebpageTool(),
                # Traditional research tools
                ComprehensiveSearchTool(),
                DataAnalysisTool(),
                CitationTool(),
                ResearchSummaryTool()
            ]
        )

    async def respond(self, user_input: str) -> str:
        """Generate response to user input.
        
        Args:
            user_input: User's question or request
            
        Returns:
            Academic research-focused response
        """
        # Add research context to the prompt
        prompt = f"""Consider the following research question or topic:

{user_input}

Provide a thorough academic analysis, including:
1. Relevant academic context
2. Methodological considerations
3. Key findings or theoretical frameworks
4. Potential research directions
5. Academic references if applicable

Response:"""

        return await self._generate_response(prompt)
