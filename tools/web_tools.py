"""Web-based tools for agents."""
import os
from typing import Optional, Dict, Any, List
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from .base import BaseTool

class WebScraperTool(BaseTool):
    """Tool for scraping web content."""
    
    def __init__(self):
        super().__init__(
            name="Web Scraper",
            description="Scrape content from web pages.",
            parameters={
                "url": {
                    "type": "string",
                    "description": "URL to scrape"
                },
                "selectors": {
                    "type": "array",
                    "description": "CSS selectors to extract content",
                    "items": {
                        "type": "string"
                    }
                }
            }
        )
    
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute web scraping."""
        url = kwargs.get("url", "")
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            content = {
                "title": soup.title.string if soup.title else "No title found",
                "meta_description": soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else "No meta description found",
                "h1_tags": [h1.text for h1 in soup.find_all('h1')],
                "main_content": soup.find('main').text if soup.find('main') else "No main content found"
            }
            
            return content
        except Exception as e:
            return {"error": str(e)}

class SitemapAnalyzerTool(BaseTool):
    """Tool for analyzing XML sitemaps."""
    
    def __init__(self):
        super().__init__(
            name="Sitemap Analyzer",
            description="Analyze XML sitemaps for structure and content.",
            parameters={
                "url": {
                    "type": "string",
                    "description": "URL of the sitemap"
                }
            }
        )
    
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute sitemap analysis."""
        url = kwargs.get("url", "")
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'xml')
            
            urls = soup.find_all('url')
            analysis = {
                "total_urls": len(urls),
                "urls": [
                    {
                        "loc": url.find('loc').text,
                        "lastmod": url.find('lastmod').text if url.find('lastmod') else None,
                        "priority": url.find('priority').text if url.find('priority') else None
                    }
                    for url in urls[:10]  # Limit to first 10 URLs
                ]
            }
            
            return analysis
        except Exception as e:
            return {"error": str(e)}

class RobotsTxtTool(BaseTool):
    """Tool for analyzing robots.txt files."""
    
    def __init__(self):
        super().__init__(
            name="Robots.txt Analyzer",
            description="Analyze robots.txt files for crawling directives.",
            parameters={
                "url": {
                    "type": "string",
                    "description": "URL of the website (robots.txt will be fetched from root)"
                }
            }
        )
    
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute robots.txt analysis."""
        url = kwargs.get("url", "")
        try:
            # Ensure URL ends with robots.txt
            if not url.endswith('robots.txt'):
                url = url.rstrip('/') + '/robots.txt'
            
            response = requests.get(url)
            response.raise_for_status()
            
            lines = response.text.split('\n')
            directives = []
            current_agent = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('User-agent:'):
                    current_agent = line.split(':', 1)[1].strip()
                    directives.append({
                        "user_agent": current_agent,
                        "rules": []
                    })
                elif line.startswith(('Allow:', 'Disallow:', 'Crawl-delay:', 'Sitemap:')):
                    if current_agent and directives:
                        rule_type, value = line.split(':', 1)
                        directives[-1]["rules"].append({
                            "type": rule_type.strip(),
                            "value": value.strip()
                        })
            
            return {
                "directives": directives,
                "raw_content": response.text
            }
        except Exception as e:
            return {"error": str(e)}

class WebSearchTool(BaseTool):
    """Tool for performing web searches."""
    
    def __init__(self):
        """Initialize the web search tool."""
        super().__init__(
            name="web_search",
            description="Search the web for information",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                }
            }
        )
        
    def run(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Perform a web search.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Dict[str, Any]: Search results
        """
        # Mock implementation for testing
        return {
            "results": [
                {
                    "title": f"Result {i} for {query}",
                    "snippet": f"Mock search result {i} for query: {query}",
                    "url": f"https://example.com/result{i}"
                }
                for i in range(num_results)
            ]
        }

class TranslationTool(BaseTool):
    """Tool for translating text between languages."""
    
    def __init__(self):
        """Initialize the translation tool."""
        super().__init__(
            name="translation",
            description="Translates text between languages",
            parameters={
                "text": {
                    "type": "string",
                    "description": "Text to translate"
                },
                "source_lang": {
                    "type": "string",
                    "description": "Source language code (e.g., 'en', 'de')",
                    "default": "auto"
                },
                "target_lang": {
                    "type": "string",
                    "description": "Target language code",
                    "default": "en"
                }
            }
        )
        
    def run(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
        """Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (default: auto-detect)
            target_lang: Target language code (default: English)
            
        Returns:
            str: Translated text
        """
        # Mock implementation for testing
        return f"Translated: {text}"

class SEOAnalysisTool(BaseTool):
    """Tool for analyzing websites for SEO."""
    
    def __init__(self):
        super().__init__(
            name="SEO Analysis",
            description="Analyze a website for SEO factors",
            parameters={
                "url": {
                    "type": "string",
                    "description": "URL to analyze"
                },
                "keywords": {
                    "type": "array",
                    "description": "Target keywords to analyze",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "depth": {
                    "type": "string",
                    "description": "Analysis depth (basic or full)",
                    "enum": ["basic", "full"],
                    "default": "basic"
                }
            }
        )
    
    def run(self, url: str, keywords: Optional[List[str]] = None, depth: str = "basic") -> Dict[str, Any]:
        """Run SEO analysis on a URL.
        
        Args:
            url: URL to analyze
            keywords: Optional list of target keywords
            depth: Analysis depth (basic or full)
            
        Returns:
            Dict containing analysis results
        """
        if not url:
            raise ValueError("URL is required for SEO analysis")
            
        keywords = keywords or []
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Basic SEO analysis
            title = soup.title.string if soup.title else "No title found"
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            meta_desc = meta_desc['content'] if meta_desc else "No meta description found"
            
            h1_tags = [h1.text for h1 in soup.find_all('h1')]
            h2_tags = [h2.text for h2 in soup.find_all('h2')]
            img_tags = soup.find_all('img')
            
            # Keyword analysis
            content = soup.get_text().lower()
            keyword_analysis = {}
            for keyword in keywords:
                keyword = keyword.lower()
                count = content.count(keyword)
                keyword_analysis[keyword] = {
                    'count': count,
                    'density': count / len(content.split()) if content else 0,
                    'in_title': keyword in title.lower() if title else False,
                    'in_meta': keyword in meta_desc.lower() if meta_desc else False,
                    'in_headings': any(keyword in h.lower() for h in h1_tags + h2_tags)
                }
            
            results = {
                "url": url,
                "title": {
                    "content": title,
                    "length": len(title) if title else 0
                },
                "meta_description": {
                    "content": meta_desc,
                    "length": len(meta_desc) if meta_desc else 0
                },
                "headings": {
                    "h1": h1_tags,
                    "h2": h2_tags
                },
                "images": {
                    "total": len(img_tags),
                    "missing_alt": len([img for img in img_tags if not img.get('alt')])
                },
                "keyword_analysis": keyword_analysis
            }
            
            if depth == "full":
                # Additional analysis for full depth
                links = soup.find_all('a')
                results.update({
                    "links": {
                        "total": len(links),
                        "internal": len([l for l in links if l.get('href', '').startswith(('/'))]),
                        "external": len([l for l in links if l.get('href', '').startswith(('http', 'https'))])
                    },
                    "structured_data": bool(soup.find_all('script', {'type': 'application/ld+json'})),
                    "mobile_viewport": bool(soup.find('meta', {'name': 'viewport'})),
                    "canonical_url": bool(soup.find('link', {'rel': 'canonical'}))
                })
            
            return results
            
        except Exception as e:
            return {"error": str(e)}
