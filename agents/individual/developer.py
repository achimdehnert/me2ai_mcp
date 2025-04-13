"""
Developer Agent implementation using Filesystem and GitHub MCP capabilities.

This agent specializes in working with codebases, repositories, and file systems.
"""
from typing import Dict, List, Any, Optional, Union
import json

from ..base import BaseAgent
from me2ai.llms.base import LLMProvider
from me2ai.tools.filesystem_mcp_tools import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    SearchFilesTool,
    GetFileInfoTool
)
from me2ai.tools.github_mcp_tools import (
    SearchRepositoriesTool,
    GetRepositoryDetailsTool,
    ListRepositoryContentsTool,
    GetFileContentTool,
    SearchCodeTool,
    ListIssuestTool,
    GetIssueDetailsTool
)

class DeveloperAgent(BaseAgent):
    """Code development, repository management, and project organization specialist."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        memory: Optional[Dict] = None,
        tools: Optional[List] = None,
        verbose: bool = False,
    ):
        """Initialize the Developer agent.
        
        Args:
            llm_provider: Language model provider to use
            memory: Memory dict to use
            tools: Additional tools to include
            verbose: Whether to enable verbose mode
        """
        super().__init__(
            name="Developer",
            system_prompt="""You are a development specialist who:
            1. Analyzes code repositories and project structures
            2. Navigates complex codebases efficiently
            3. Works with GitHub repositories and issues
            4. Manages file systems and code organization
            5. Searches and finds relevant code patterns
            
            You can use tools to:
            - Read and write to files
            - List directory contents and search for files
            - Search GitHub repositories and code
            - Explore repository contents and issues
            - Get detailed information about files and repositories
            
            Focus on providing accurate, relevant code analysis and assistance with project organization.
            Your MCP-powered toolset allows you to work with both local filesystems and GitHub repositories efficiently.""",
            llm_provider=llm_provider,
            memory=memory,
            tools=tools or [
                # Filesystem tools
                ReadFileTool(),
                WriteFileTool(),
                ListDirectoryTool(),
                SearchFilesTool(),
                GetFileInfoTool(),
                
                # GitHub tools
                SearchRepositoriesTool(),
                GetRepositoryDetailsTool(),
                ListRepositoryContentsTool(),
                GetFileContentTool(),
                SearchCodeTool(),
                ListIssuestTool(),
                GetIssueDetailsTool()
            ],
            verbose=verbose,
        )
