"""Web scraping specialist agent."""
from typing import Optional
from langchain_core.memory import BaseMemory
from ..base import BaseAgent
from me2ai.llms.base import LLMProvider
from me2ai.tools.web_tools import WebScraperTool, SitemapAnalyzerTool, RobotsTxtTool
from me2ai.tools.mcp_tools import (
    FetchWebpageTool,
    ExtractElementsTool,
    SummarizeWebpageTool
)

class WebScraperAgent(BaseAgent):
    """Web scraping and data collection specialist."""
    
    def __init__(self, llm_provider: LLMProvider, memory: Optional[BaseMemory] = None):
        """Initialize the web scraper agent.
        
        Args:
            llm_provider: The LLM provider to use
            memory: Optional memory instance
        """
        super().__init__(
            role="Web Scraper",
            system_prompt="""You are a web scraping specialist who:
            1. Extracts structured data from websites
            2. Analyzes site architecture and crawlability
            3. Handles different data formats and APIs
            4. Respects robots.txt and site policies
            5. Summarizes web content effectively
            
            You can use tools to:
            - Fetch and process web content efficiently
            - Extract specific elements using CSS selectors
            - Summarize webpage content automatically
            - Analyze sitemaps for structure
            - Check robots.txt compliance
            - Parse various data formats
            
            Your improved MCP-powered toolset allows for more accurate and efficient web scraping with fewer errors.
            Focus on collecting comprehensive, accurate data while following web scraping best practices.""",
            llm_provider=llm_provider,
            memory=memory,
            tools=[
                # MCP-powered tools
                FetchWebpageTool(),
                ExtractElementsTool(),
                SummarizeWebpageTool(),
                # Standard tools
                WebScraperTool(),
                SitemapAnalyzerTool(),
                RobotsTxtTool()
            ]
        )
