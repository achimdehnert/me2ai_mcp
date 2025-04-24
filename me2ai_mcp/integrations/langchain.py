"""
LangChain integration for ME2AI MCP.

This module provides adapters and utilities for integrating
ME2AI MCP tools with LangChain.
"""

from typing import Any, Dict, List, Optional, Union, Type, Callable, Awaitable
import logging
import inspect
import json
from functools import wraps
import asyncio

# Import LangChain
from langchain.tools import BaseTool as LCBaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field

# Import ME2AI MCP base classes
from me2ai_mcp.base import BaseTool


class MCPToolArguments(BaseModel):
    """Base model for MCP tool arguments."""
    
    # This will be extended with runtime fields
    pass


class LangChainToolAdapter(LCBaseTool):
    """Adapter for using ME2AI MCP tools with LangChain.
    
    This class wraps ME2AI MCP tools to make them compatible with
    the LangChain tool interface.
    
    Attributes:
        mcp_tool: The ME2AI MCP tool being wrapped
        name: Tool name
        description: Tool description
        args_schema: Pydantic schema for arguments
        logger: Logger instance
    """
    
    def __init__(
        self,
        mcp_tool: BaseTool,
        name: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[Type[BaseModel]] = None,
        return_direct: bool = False,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """Initialize the LangChain tool adapter.
        
        Args:
            mcp_tool: The ME2AI MCP tool to adapt
            name: Override name for the tool
            description: Override description for the tool
            args_schema: Custom args schema for the tool
            return_direct: Whether to return the output directly
            verbose: Whether to enable verbose logging
            **kwargs: Additional kwargs for LangChain tool
        """
        self.mcp_tool = mcp_tool
        
        # Use MCP tool name and description if not provided
        tool_name = name or mcp_tool.name
        tool_description = description or mcp_tool.description
        
        # Set up args schema if not provided
        if args_schema is None:
            args_schema = self._create_args_schema()
        
        # Initialize LangChain tool
        super().__init__(
            name=tool_name,
            description=tool_description,
            args_schema=args_schema,
            return_direct=return_direct,
            verbose=verbose,
            **kwargs
        )
        
        # Set up logging
        self.logger = logging.getLogger(f"me2ai-mcp-langchain-{tool_name}")
    
    def _create_args_schema(self) -> Type[BaseModel]:
        """Create a Pydantic schema for the tool arguments.
        
        Returns:
            Type[BaseModel]: Pydantic schema for arguments
        """
        # Get argument information from the MCP tool's execute method
        try:
            signature = inspect.signature(self.mcp_tool.execute)
            parameters = {}
            
            # Exclude 'self' parameter
            for name, param in signature.parameters.items():
                if name == 'self':
                    continue
                
                # Determine field type and default value
                annotation = param.annotation
                if annotation is inspect.Parameter.empty:
                    annotation = Any
                
                # Handle default value
                if param.default is inspect.Parameter.empty:
                    # Required parameter
                    field = Field(..., description=f"Parameter {name}")
                else:
                    # Optional parameter
                    field = Field(default=param.default, description=f"Parameter {name}")
                
                parameters[name] = (annotation, field)
            
            # Create a dynamic Pydantic model
            args_schema = type(
                f"{self.mcp_tool.name.capitalize()}Arguments",
                (MCPToolArguments,),
                parameters
            )
            
            return args_schema
            
        except Exception as e:
            self.logger.warning(f"Could not create args schema: {e}")
            
            # Use base schema instead
            return MCPToolArguments
    
    def _run(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Run the MCP tool with the provided arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Any: Result from the MCP tool
            
        Raises:
            Exception: If tool execution fails
        """
        # Handle both *args and **kwargs
        if args and len(args) == 1 and isinstance(args[0], str):
            # Single string argument - try to parse as JSON
            try:
                parsed_args = json.loads(args[0])
                if isinstance(parsed_args, dict):
                    kwargs.update(parsed_args)
            except json.JSONDecodeError:
                # Not JSON, treat as first parameter if appropriate
                arg_schema = self.args_schema and getattr(self.args_schema, "__annotations__", {})
                if arg_schema:
                    first_arg_name = next(iter(arg_schema), None)
                    if first_arg_name:
                        kwargs[first_arg_name] = args[0]
        
        try:
            # Execute the MCP tool
            self.logger.info(f"Executing {self.name} with params: {kwargs}")
            result = asyncio.run(self.mcp_tool.execute(**kwargs))
            
            # Format the result to match LangChain expectations
            if isinstance(result, dict) and "success" in result:
                if not result["success"]:
                    raise Exception(result.get("error", "Tool execution failed"))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {str(e)}")
            raise
    
    async def _arun(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Run the MCP tool asynchronously.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Any: Result from the MCP tool
            
        Raises:
            Exception: If tool execution fails
        """
        # Handle both *args and **kwargs
        if args and len(args) == 1 and isinstance(args[0], str):
            # Single string argument - try to parse as JSON
            try:
                parsed_args = json.loads(args[0])
                if isinstance(parsed_args, dict):
                    kwargs.update(parsed_args)
            except json.JSONDecodeError:
                # Not JSON, treat as first parameter if appropriate
                arg_schema = self.args_schema and getattr(self.args_schema, "__annotations__", {})
                if arg_schema:
                    first_arg_name = next(iter(arg_schema), None)
                    if first_arg_name:
                        kwargs[first_arg_name] = args[0]
        
        try:
            # Execute the MCP tool
            self.logger.info(f"Executing {self.name} with params: {kwargs}")
            result = await self.mcp_tool.execute(**kwargs)
            
            # Format the result to match LangChain expectations
            if isinstance(result, dict) and "success" in result:
                if not result["success"]:
                    raise Exception(result.get("error", "Tool execution failed"))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {str(e)}")
            raise


def create_langchain_tools(
    mcp_tools: List[BaseTool],
    return_direct: bool = False,
    verbose: bool = False
) -> List[LangChainToolAdapter]:
    """Create LangChain tools from ME2AI MCP tools.
    
    Args:
        mcp_tools: List of MCP tools to adapt
        return_direct: Whether to return the output directly
        verbose: Whether to enable verbose logging
        
    Returns:
        List[LangChainToolAdapter]: List of LangChain tools
    """
    return [
        LangChainToolAdapter(
            mcp_tool=tool,
            return_direct=return_direct,
            verbose=verbose
        )
        for tool in mcp_tools
    ]


class LangChainToolFactory:
    """Factory for creating common LangChain tools.
    
    This class provides methods for creating commonly used
    LangChain tools based on ME2AI MCP tools.
    """
    
    @staticmethod
    def create_postgres_tools() -> List[LangChainToolAdapter]:
        """Create PostgreSQL database tools.
        
        Returns:
            List[LangChainToolAdapter]: List of PostgreSQL tools
        """
        from me2ai_mcp.tools.postgres import (
            ExecuteQueryTool,
            ListTablesTool,
            GetTableColumnsTool,
            GetPLZDetailsTool
        )
        
        # Create MCP tools
        mcp_tools = [
            ExecuteQueryTool(),
            ListTablesTool(),
            GetTableColumnsTool(),
            GetPLZDetailsTool()
        ]
        
        # Convert to LangChain tools
        return create_langchain_tools(mcp_tools)
    
    @staticmethod
    def create_mysql_tools() -> List[LangChainToolAdapter]:
        """Create MySQL database tools.
        
        Returns:
            List[LangChainToolAdapter]: List of MySQL tools
        """
        try:
            from me2ai_mcp.tools.mysql import (
                ExecuteQueryTool,
                ListTablesTool,
                GetTableColumnsTool,
                GetDatabaseInfoTool
            )
            
            # Create MCP tools
            mcp_tools = [
                ExecuteQueryTool(),
                ListTablesTool(),
                GetTableColumnsTool(),
                GetDatabaseInfoTool()
            ]
            
            # Convert to LangChain tools
            return create_langchain_tools(mcp_tools)
            
        except ImportError:
            logging.warning("Could not import MySQL tools, returning empty list")
            return []
    
    @staticmethod
    def create_database_tools(
        db_type: str = "postgres"
    ) -> List[LangChainToolAdapter]:
        """Create database tools based on the specified type.
        
        Args:
            db_type: Database type ("postgres" or "mysql")
            
        Returns:
            List[LangChainToolAdapter]: List of database tools
        """
        if db_type.lower() == "mysql":
            return LangChainToolFactory.create_mysql_tools()
        else:
            # Default to PostgreSQL
            return LangChainToolFactory.create_postgres_tools()
    
    @staticmethod
    def create_firecrawl_tools() -> List[LangChainToolAdapter]:
        """Create advanced web scraping tools using FireCrawl.
        
        Returns:
            List[LangChainToolAdapter]: List of FireCrawl tools for advanced web scraping
        """
        try:
            from me2ai_mcp.tools.firecrawl import (
                FireCrawlTool,
                WebContentTool,
                create_firecrawl_tool
            )
            
            # Create MCP tools
            mcp_tools = [
                # Default configuration
                FireCrawlTool(),
                # Alias as WebContentTool for compatibility
                WebContentTool()
            ]
            
            # Convert to LangChain tools
            return create_langchain_tools(mcp_tools)
            
        except ImportError:
            logging.warning("Could not import FireCrawl tools, returning empty list")
            return []

    @staticmethod
    def create_web_tools() -> List[LangChainToolAdapter]:
        """Create web fetching and search tools.
        
        Returns:
            List[LangChainToolAdapter]: List of web tools
        """
        tools = []
        
        # Try to import basic web tools
        try:
            from me2ai_mcp.tools.web import (
                WebFetchTool,
                HTMLParserTool,
                URLUtilsTool
            )
            
            # Create MCP tools
            basic_tools = [
                WebFetchTool(),
                HTMLParserTool(),
                URLUtilsTool()
            ]
            
            # Add tools to list
            tools.extend(create_langchain_tools(basic_tools))
            
        except ImportError:
            logging.warning("Could not import basic web tools")
        
        # Try to add FireCrawl tools if available
        try:
            tools.extend(LangChainToolFactory.create_firecrawl_tools())
        except Exception as e:
            logging.warning(f"Could not import FireCrawl tools: {str(e)}")
        
        return tools
    
    @staticmethod
    def create_filesystem_tools() -> List[LangChainToolAdapter]:
        """Create filesystem tools.
        
        Returns:
            List[LangChainToolAdapter]: List of filesystem tools
        """
        try:
            from me2ai_mcp.tools.filesystem import (
                ReadFileTool,
                WriteFileTool,
                ListDirectoryTool,
                FileExistsTool
            )
            
            # Create MCP tools
            mcp_tools = [
                ReadFileTool(),
                WriteFileTool(),
                ListDirectoryTool(),
                FileExistsTool()
            ]
            
            # Convert to LangChain tools
            return create_langchain_tools(mcp_tools)
            
        except ImportError:
            logging.warning("Could not import filesystem tools, returning empty list")
            return []
    
    @staticmethod
    def create_github_tools() -> List[LangChainToolAdapter]:
        """Create GitHub tools.
        
        Returns:
            List[LangChainToolAdapter]: List of GitHub tools
        """
        try:
            from me2ai_mcp.tools.github import (
                SearchRepositoriesTool,
                GetRepositoryContentTool,
                GetFileContentTool,
                SearchCodeTool
            )
            
            # Create MCP tools
            mcp_tools = [
                SearchRepositoriesTool(),
                GetRepositoryContentTool(),
                GetFileContentTool(),
                SearchCodeTool()
            ]
            
            # Convert to LangChain tools
            return create_langchain_tools(mcp_tools)
            
        except ImportError:
            logging.warning("Could not import GitHub tools, returning empty list")
            return []
