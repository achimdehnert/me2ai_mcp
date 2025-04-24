"""
Tools for ME2AI MCP servers.

This package provides reusable tool implementations for common operations in MCP servers.
"""
from .web import WebFetchTool, HTMLParserTool, URLUtilsTool
from .filesystem import FileReaderTool, FileWriterTool, DirectoryListerTool
from .github import GitHubRepositoryTool, GitHubCodeTool, GitHubIssuesTool

# Import FireCrawl tools if available
try:
    from .firecrawl import FireCrawlTool, WebContentTool, create_firecrawl_tool
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False

__all__ = [
    # Web tools
    "WebFetchTool",
    "HTMLParserTool",
    "URLUtilsTool",
    
    # Filesystem tools
    "FileReaderTool",
    "FileWriterTool",
    "DirectoryListerTool",
    
    # GitHub tools
    "GitHubRepositoryTool",
    "GitHubCodeTool",
    "GitHubIssuesTool"
]

# Add FireCrawl tools to __all__ if available
if FIRECRAWL_AVAILABLE:
    __all__.extend([
        # FireCrawl tools
        "FireCrawlTool",
        "WebContentTool", 
        "create_firecrawl_tool"
    ])
