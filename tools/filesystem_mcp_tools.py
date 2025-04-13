"""
Filesystem MCP tools for ME2AI agents.

This module provides tool interfaces for the Filesystem MCP server.
"""
from typing import Dict, List, Any, Optional, Union
import json
import os
import requests
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

class ReadFileTool(BaseTool):
    """Tool for reading file contents using the Filesystem MCP server."""
    
    name = "read_file"
    description = """Read the contents of a file. 
    Useful for examining code, data files, or any text file.
    Input should be a string representing the full file path."""
    
    def _run(self, file_path: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run the tool to read a file."""
        try:
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers={"Content-Type": "application/json"},
                json={
                    "serverName": "filesystem",
                    "toolName": "read_file",
                    "params": {"file_path": file_path}
                }
            )
            
            if response.status_code != 200:
                return f"Error: Unable to read file (Status code: {response.status_code})"
                
            result = response.json()
            
            if not result.get("success", False):
                return f"Error: {result.get('error', 'Unknown error')}"
                
            return result.get("content", "")
            
        except Exception as e:
            return f"Error: {str(e)}"

class WriteFileTool(BaseTool):
    """Tool for writing content to files using the Filesystem MCP server."""
    
    name = "write_file"
    description = """Write content to a file.
    Useful for saving data, creating scripts, or modifying configuration files.
    Input should be a JSON string with 'file_path', 'content', and optional 'overwrite' (boolean) fields."""
    
    def _run(self, tool_input: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run the tool to write a file."""
        try:
            # Parse input
            input_data = json.loads(tool_input)
            file_path = input_data.get("file_path")
            content = input_data.get("content")
            overwrite = input_data.get("overwrite", False)
            
            if not file_path or content is None:
                return "Error: Required parameters 'file_path' and 'content' must be provided"
                
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers={"Content-Type": "application/json"},
                json={
                    "serverName": "filesystem",
                    "toolName": "write_file",
                    "params": {
                        "file_path": file_path,
                        "content": content,
                        "overwrite": overwrite
                    }
                }
            )
            
            if response.status_code != 200:
                return f"Error: Unable to write file (Status code: {response.status_code})"
                
            result = response.json()
            
            if not result.get("success", False):
                return f"Error: {result.get('error', 'Unknown error')}"
                
            return f"Successfully wrote to file: {file_path}"
            
        except json.JSONDecodeError:
            return "Error: Input must be a valid JSON string"
        except Exception as e:
            return f"Error: {str(e)}"

class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents using the Filesystem MCP server."""
    
    name = "list_directory"
    description = """List the contents of a directory.
    Useful for exploring file systems, finding files, or understanding project structure.
    Input should be a string representing the directory path, or a JSON string with 'directory_path' and optional 'pattern' fields."""
    
    def _run(self, tool_input: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run the tool to list directory contents."""
        try:
            # Parse input - handle both string and JSON
            pattern = None
            
            if tool_input.startswith("{") and tool_input.endswith("}"):
                try:
                    input_data = json.loads(tool_input)
                    directory_path = input_data.get("directory_path")
                    pattern = input_data.get("pattern")
                except:
                    directory_path = tool_input
            else:
                directory_path = tool_input
                
            if not directory_path:
                return "Error: Directory path must be provided"
                
            params = {"directory_path": directory_path}
            if pattern:
                params["pattern"] = pattern
                
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers={"Content-Type": "application/json"},
                json={
                    "serverName": "filesystem",
                    "toolName": "list_directory",
                    "params": params
                }
            )
            
            if response.status_code != 200:
                return f"Error: Unable to list directory (Status code: {response.status_code})"
                
            result = response.json()
            
            if not result.get("success", False):
                return f"Error: {result.get('error', 'Unknown error')}"
                
            # Format the results as a readable string
            items = result.get("items", [])
            output = f"Directory: {result.get('directory')}\n"
            output += f"Total items: {len(items)}\n\n"
            
            files = [item for item in items if item.get("type") == "file"]
            directories = [item for item in items if item.get("type") == "directory"]
            
            output += "Directories:\n"
            for dir_item in sorted(directories, key=lambda x: x.get("name", "")):
                output += f"- {dir_item.get('name')}/\n"
                
            output += "\nFiles:\n"
            for file_item in sorted(files, key=lambda x: x.get("name", "")):
                size = file_item.get("size", 0)
                size_str = f"{size} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                output += f"- {file_item.get('name')} ({size_str})\n"
                
            return output
            
        except Exception as e:
            return f"Error: {str(e)}"

class SearchFilesTool(BaseTool):
    """Tool for searching file contents using the Filesystem MCP server."""
    
    name = "search_files"
    description = """Search for files containing specific text.
    Useful for finding code implementations, references, or data.
    Input should be a JSON string with 'directory_path', 'query', and optional 'file_extensions', 'recursive', and 'max_results' fields."""
    
    def _run(self, tool_input: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run the tool to search files."""
        try:
            # Parse input
            input_data = json.loads(tool_input)
            directory_path = input_data.get("directory_path")
            query = input_data.get("query")
            file_extensions = input_data.get("file_extensions")
            recursive = input_data.get("recursive", True)
            max_results = input_data.get("max_results", 50)
            
            if not directory_path or not query:
                return "Error: Required parameters 'directory_path' and 'query' must be provided"
                
            params = {
                "directory_path": directory_path,
                "query": query,
                "recursive": recursive,
                "max_results": max_results
            }
            
            if file_extensions:
                params["file_extensions"] = file_extensions
                
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers={"Content-Type": "application/json"},
                json={
                    "serverName": "filesystem",
                    "toolName": "search_files",
                    "params": params
                }
            )
            
            if response.status_code != 200:
                return f"Error: Unable to search files (Status code: {response.status_code})"
                
            result = response.json()
            
            if not result.get("success", False):
                return f"Error: {result.get('error', 'Unknown error')}"
                
            # Format the results as a readable string
            matches = result.get("matches", [])
            output = f"Search query: \"{result.get('query')}\"\n"
            output += f"Matches found: {len(matches)}\n\n"
            
            if result.get("max_reached", False):
                output += f"Note: Maximum result limit reached ({max_results}).\n\n"
                
            for i, match in enumerate(matches, 1):
                output += f"{i}. {match.get('path')}\n"
                
                # Add sample matching lines
                matching_lines = match.get("matching_lines", [])
                if matching_lines:
                    output += "   Matching lines:\n"
                    for line_info in matching_lines:
                        line_num = line_info.get("line_number")
                        content = line_info.get("content", "").strip()
                        output += f"   - Line {line_num}: {content}\n"
                        
                output += "\n"
                
            return output
            
        except json.JSONDecodeError:
            return "Error: Input must be a valid JSON string"
        except Exception as e:
            return f"Error: {str(e)}"

class GetFileInfoTool(BaseTool):
    """Tool for getting detailed file information using the Filesystem MCP server."""
    
    name = "get_file_info"
    description = """Get detailed information about a file or directory.
    Useful for checking file metadata, permissions, or existence.
    Input should be a string representing the file path."""
    
    def _run(self, file_path: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run the tool to get file information."""
        try:
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers={"Content-Type": "application/json"},
                json={
                    "serverName": "filesystem",
                    "toolName": "get_file_info",
                    "params": {"file_path": file_path}
                }
            )
            
            if response.status_code != 200:
                return f"Error: Unable to get file info (Status code: {response.status_code})"
                
            result = response.json()
            
            if not result.get("success", False):
                return f"Error: {result.get('error', 'Unknown error')}"
                
            # Format the results as a readable string
            info = result.get("info", {})
            
            output = f"File information for: {info.get('path')}\n"
            output += f"Type: {info.get('type')}\n"
            output += f"Name: {info.get('name')}\n"
            
            size = info.get("size", 0)
            size_str = f"{size} bytes" if size < 1024 else f"{size/1024:.1f} KB"
            output += f"Size: {size_str}\n"
            
            output += f"Created: {info.get('created')}\n"
            output += f"Modified: {info.get('modified')}\n"
            
            if info.get("type") == "file":
                output += f"Extension: {info.get('extension', '')}\n"
                output += f"Is text file: {info.get('is_text', False)}\n"
                
            return output
            
        except Exception as e:
            return f"Error: {str(e)}"
