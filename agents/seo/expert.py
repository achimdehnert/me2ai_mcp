"""SEO Expert agent implementation."""
from typing import Dict, List, Optional
from ..base import BaseAgent
from ...tools.seo.technical_tools import TechnicalSEOTools
from ...tools.seo.local_tools import LocalSEOTools
from ...tools.seo.ecommerce_tools import EcommerceSEOTools
from ...tools.seo.reporting_tools import ReportingTools
from ...memory import BaseMemory
from ...llms.base import LLMProvider

class SEOExpertAgent(BaseAgent):
    """SEO Expert agent that provides comprehensive SEO analysis and recommendations."""

    DEFAULT_PROMPT = """You are an experienced SEO expert with deep knowledge of technical SEO, 
    content optimization, and search engine algorithms. Your role is to analyze websites and provide 
    actionable recommendations to improve their search engine visibility and performance."""

    def __init__(self, llm_provider: LLMProvider, memory: Optional[BaseMemory] = None):
        """Initialize SEO Expert agent with specialized tools."""
        super().__init__(
            role="SEO Expert",
            system_prompt=self.DEFAULT_PROMPT,
            llm_provider=llm_provider,
            memory=memory
        )
        self.tools = {
            "technical_seo": TechnicalSEOTools(),
            "local_seo": LocalSEOTools(),
            "ecommerce_seo": EcommerceSEOTools(),
            "reporting": ReportingTools()
        }

    def analyze_technical_seo(self, url: str) -> Dict:
        """Perform technical SEO analysis of a URL."""
        return self.tools["technical_seo"].run(url)

    def analyze_local_seo(self, business_info: Dict) -> Dict:
        """Analyze local SEO presence."""
        return self.tools["local_seo"].run(business_info)

    def analyze_ecommerce_seo(self, store_url: str) -> Dict:
        """Analyze e-commerce SEO performance."""
        return self.tools["ecommerce_seo"].run(store_url)

    def generate_report(self, data: Dict) -> str:
        """Generate SEO analysis report."""
        return self.tools["reporting"].run(data)

    async def respond(self, message: str) -> str:
        """Generate a response to the user's message.
        
        Args:
            message: User message
            
        Returns:
            str: Agent response
        """
        messages = self._build_messages(message)
        response = await self.llm_provider.generate_response(messages)
        return response
