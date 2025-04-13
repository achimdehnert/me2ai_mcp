"""
Utility functions for ME2AI MCP servers.

This module provides common utility functions for processing inputs,
formatting responses, and handling data in MCP servers.
"""
from typing import Dict, List, Any, Optional, Union
import re
import json
import logging
from datetime import datetime
import textwrap

# Optional dependencies
try:
    import bleach
    BLEACH_AVAILABLE = True
except ImportError:
    BLEACH_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# Configure logging
logger = logging.getLogger("me2ai-mcp-utils")


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """Sanitize text input for safe processing.
    
    Args:
        text: Input text
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    # Check input type
    if not isinstance(text, str):
        logger.warning(f"Expected string input, got {type(text)}")
        text = str(text)
    
    # Truncate to maximum length
    if len(text) > max_length:
        logger.warning(f"Input truncated from {len(text)} to {max_length} characters")
        text = text[:max_length]
    
    # Remove null bytes and other control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Use bleach for HTML sanitization if available
    if BLEACH_AVAILABLE:
        text = bleach.clean(text, strip=True)
    
    return text


def format_response(
    data: Any,
    format_type: str = "auto",
    success: bool = True,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """Format a standard MCP response.
    
    Args:
        data: Response data
        format_type: Response format type (auto, text, json, html)
        success: Whether the operation was successful
        error: Error message if unsuccessful
        
    Returns:
        Formatted response dictionary
    """
    # Basic response structure
    response = {
        "success": success,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add error message if provided
    if error:
        response["error"] = error
        response["success"] = False
    
    # Auto-detect format if needed
    if format_type == "auto":
        if isinstance(data, dict) or isinstance(data, list):
            format_type = "json"
        elif isinstance(data, str) and data.strip().startswith(("<html", "<!DOCTYPE")):
            format_type = "html"
        else:
            format_type = "text"
    
    # Format data based on type
    if format_type == "json":
        response["format"] = "json"
        response["content"] = data
    elif format_type == "html":
        response["format"] = "html"
        response["content"] = data
    else:  # Default to text
        response["format"] = "text"
        response["content"] = str(data)
    
    # Add format-specific metadata
    if format_type == "text":
        response["content_length"] = len(response["content"])
        lines = response["content"].split("\n")
        response["line_count"] = len(lines)
    
    return response


def extract_text(
    html_content: str,
    max_length: int = 10000,
    include_headings: bool = True,
    include_links: bool = True
) -> str:
    """Extract readable text from HTML content.
    
    Args:
        html_content: HTML content
        max_length: Maximum length of extracted text
        include_headings: Whether to include headings
        include_links: Whether to include link information
        
    Returns:
        Extracted text
    """
    # Check if BeautifulSoup is available
    if not BS4_AVAILABLE:
        logger.warning("BeautifulSoup not available, falling back to basic HTML removal")
        # Basic HTML tag removal with regex
        text = re.sub(r'<[^>]*>', ' ', html_content)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_length]
    
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for element in soup(["script", "style", "noscript", "iframe", "footer"]):
        element.extract()
    
    # Get all text
    text = soup.get_text(separator='\n', strip=True)
    
    # Add headings with emphasis if requested
    if include_headings:
        headings = []
        for tag in ['h1', 'h2', 'h3', 'h4']:
            for heading in soup.find_all(tag):
                heading_text = heading.get_text(strip=True)
                if heading_text:
                    level = int(tag[1])
                    prefix = "#" * level
                    headings.append(f"{prefix} {heading_text}")
        
        if headings:
            text = '\n\n'.join(headings) + '\n\n' + text
    
    # Add link information if requested
    if include_links:
        links = []
        for link in soup.find_all('a', href=True):
            link_text = link.get_text(strip=True)
            href = link['href']
            if link_text and href and not href.startswith('#') and not href.startswith('javascript:'):
                links.append(f"- [{link_text}]({href})")
        
        if links:
            text += '\n\n### Links:\n' + '\n'.join(links)
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Truncate if necessary
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text


def summarize_text(text: str, max_length: int = 1000, preserve_sentences: bool = True) -> str:
    """Summarize text by truncation with sentence preservation.
    
    Args:
        text: Text to summarize
        max_length: Maximum length of summary
        preserve_sentences: Whether to preserve complete sentences
        
    Returns:
        Summarized text
    """
    if len(text) <= max_length:
        return text
    
    if preserve_sentences:
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Add sentences until we reach the max length
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) + 1 <= max_length:
                summary += sentence + " "
            else:
                break
                
        return summary.strip()
    else:
        # Simple truncation
        return text[:max_length] + "..."


def wrap_text_block(text: str, width: int = 80, prefix: str = "") -> str:
    """Wrap text to a specified width with optional prefix.
    
    Args:
        text: Text to wrap
        width: Line width
        prefix: Prefix to add to each line
        
    Returns:
        Wrapped text
    """
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=prefix,
        subsequent_indent=prefix
    )
    
    lines = text.split('\n')
    wrapped_lines = [wrapper.fill(line) if line.strip() else '' for line in lines]
    
    return '\n'.join(wrapped_lines)
