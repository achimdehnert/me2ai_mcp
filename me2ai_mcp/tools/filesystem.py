"""
Filesystem tools for ME2AI MCP servers.

This module provides common tools for file and directory operations
that can be used across different MCP servers.
"""
from typing import Dict, List, Any, Optional
import os
import logging
from dataclasses import dataclass
from pathlib import Path
import glob
from ..base import BaseTool

# Configure logging
logger = logging.getLogger("me2ai-mcp-tools-filesystem")


@dataclass
class FileReaderTool(BaseTool):
    """Tool for reading file content."""
    
    name: str = "read_file"
    description: str = "Read content from a file"
    max_file_size: int = 1024 * 1024 * 5  # 5MB
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read a file and return its content.
        
        Args:
            params: Dictionary containing:
                - file_path: Path to the file to read
                - encoding: File encoding (default: utf-8)
                - binary: Whether to read as binary (default: False)
        
        Returns:
            Dictionary containing file content and metadata
        """
        file_path = params.get("file_path")
        if not file_path:
            return {
                "success": False,
                "error": "file_path parameter is required"
            }
            
        encoding = params.get("encoding", "utf-8")
        binary = params.get("binary", False)
        
        try:
            # Normalize path
            file_path = os.path.abspath(file_path)
            
            # Check if file exists
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
                
            # Check if path is a file
            if not os.path.isfile(file_path):
                return {
                    "success": False,
                    "error": f"Path is not a file: {file_path}"
                }
                
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                return {
                    "success": False,
                    "error": f"File too large: {file_size} bytes (max {self.max_file_size})"
                }
                
            # Read file content
            if binary:
                with open(file_path, "rb") as f:
                    content = f.read()
                    # Convert binary to base64 for JSON compatibility
                    import base64
                    content = base64.b64encode(content).decode("utf-8")
            else:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                    
            # Get file stats
            stats = os.stat(file_path)
            
            # Return results
            return {
                "success": True,
                "file_path": file_path,
                "content": content,
                "size": file_size,
                "encoding": encoding if not binary else None,
                "binary": binary,
                "metadata": {
                    "created": stats.st_ctime,
                    "modified": stats.st_mtime,
                    "accessed": stats.st_atime,
                    "extension": os.path.splitext(file_path)[1],
                    "filename": os.path.basename(file_path)
                }
            }
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Error reading file: {str(e)}",
                "exception_type": type(e).__name__
            }


@dataclass
class FileWriterTool(BaseTool):
    """Tool for writing content to files."""
    
    name: str = "write_file"
    description: str = "Write content to a file"
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Write content to a file.
        
        Args:
            params: Dictionary containing:
                - file_path: Path to the file to write
                - content: Content to write
                - encoding: File encoding (default: utf-8)
                - binary: Whether content is binary (base64-encoded) (default: False)
                - overwrite: Whether to overwrite existing file (default: False)
                - append: Whether to append to existing file (default: False)
        
        Returns:
            Dictionary containing operation result
        """
        file_path = params.get("file_path")
        content = params.get("content")
        
        if not file_path:
            return {
                "success": False,
                "error": "file_path parameter is required"
            }
            
        if content is None:
            return {
                "success": False,
                "error": "content parameter is required"
            }
            
        encoding = params.get("encoding", "utf-8")
        binary = params.get("binary", False)
        overwrite = params.get("overwrite", False)
        append = params.get("append", False)
        
        try:
            # Normalize path
            file_path = os.path.abspath(file_path)
            
            # Check if file exists
            file_exists = os.path.exists(file_path)
            
            if file_exists and not (overwrite or append):
                return {
                    "success": False,
                    "error": f"File already exists: {file_path} (set overwrite=True to replace or append=True to add content)"
                }
                
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write content
            mode = "wb" if binary else "w"
            if append:
                mode = "ab" if binary else "a"
                
            if binary:
                # Decode base64 content
                import base64
                binary_content = base64.b64decode(content)
                with open(file_path, mode) as f:
                    f.write(binary_content)
            else:
                with open(file_path, mode, encoding=encoding) as f:
                    f.write(content)
                    
            # Get file stats
            stats = os.stat(file_path)
            
            # Return results
            return {
                "success": True,
                "file_path": file_path,
                "size": os.path.getsize(file_path),
                "operation": "append" if append else ("overwrite" if file_exists else "create"),
                "metadata": {
                    "created": stats.st_ctime,
                    "modified": stats.st_mtime,
                    "extension": os.path.splitext(file_path)[1],
                    "filename": os.path.basename(file_path)
                }
            }
            
        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Error writing to file: {str(e)}",
                "exception_type": type(e).__name__
            }


@dataclass
class DirectoryListerTool(BaseTool):
    """Tool for listing directory contents."""
    
    name: str = "list_directory"
    description: str = "List contents of a directory"
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List the contents of a directory.
        
        Args:
            params: Dictionary containing:
                - directory_path: Path to the directory to list
                - pattern: Optional glob pattern to filter results
                - recursive: Whether to list subdirectories recursively (default: False)
                - include_hidden: Whether to include hidden files (default: False)
                - max_depth: Maximum recursion depth (default: 1)
        
        Returns:
            Dictionary containing directory contents
        """
        directory_path = params.get("directory_path")
        if not directory_path:
            return {
                "success": False,
                "error": "directory_path parameter is required"
            }
            
        pattern = params.get("pattern")
        recursive = params.get("recursive", False)
        include_hidden = params.get("include_hidden", False)
        max_depth = params.get("max_depth", 1)
        
        try:
            # Normalize path
            directory_path = os.path.abspath(directory_path)
            
            # Check if directory exists
            if not os.path.exists(directory_path):
                return {
                    "success": False,
                    "error": f"Directory not found: {directory_path}"
                }
                
            # Check if path is a directory
            if not os.path.isdir(directory_path):
                return {
                    "success": False,
                    "error": f"Path is not a directory: {directory_path}"
                }
                
            # List directory contents
            items = []
            
            if recursive:
                # Recursive listing with max_depth control
                for root, dirs, files in os.walk(directory_path):
                    # Calculate current depth
                    depth = root[len(directory_path):].count(os.sep)
                    if depth > max_depth - 1:
                        continue
                        
                    # Skip hidden directories if not included
                    if not include_hidden:
                        dirs[:] = [d for d in dirs if not d.startswith(".")]
                        
                    # Process files
                    for file in files:
                        # Skip hidden files if not included
                        if not include_hidden and file.startswith("."):
                            continue
                            
                        file_path = os.path.join(root, file)
                        
                        # Apply pattern filter if specified
                        if pattern and not glob.fnmatch.fnmatch(file, pattern):
                            continue
                            
                        # Add file info
                        stats = os.stat(file_path)
                        items.append({
                            "name": file,
                            "path": file_path,
                            "type": "file",
                            "size": stats.st_size,
                            "created": stats.st_ctime,
                            "modified": stats.st_mtime,
                            "relative_path": os.path.relpath(file_path, directory_path)
                        })
                        
                    # Add directory info
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        
                        # Skip hidden directories if not included
                        if not include_hidden and dir_name.startswith("."):
                            continue
                            
                        stats = os.stat(dir_path)
                        items.append({
                            "name": dir_name,
                            "path": dir_path,
                            "type": "directory",
                            "size": None,
                            "created": stats.st_ctime,
                            "modified": stats.st_mtime,
                            "relative_path": os.path.relpath(dir_path, directory_path)
                        })
            else:
                # Non-recursive listing
                for item in os.listdir(directory_path):
                    # Skip hidden items if not included
                    if not include_hidden and item.startswith("."):
                        continue
                        
                    item_path = os.path.join(directory_path, item)
                    
                    # Apply pattern filter for files
                    if pattern and os.path.isfile(item_path) and not glob.fnmatch.fnmatch(item, pattern):
                        continue
                        
                    # Add item info
                    stats = os.stat(item_path)
                    is_dir = os.path.isdir(item_path)
                    
                    items.append({
                        "name": item,
                        "path": item_path,
                        "type": "directory" if is_dir else "file",
                        "size": None if is_dir else stats.st_size,
                        "created": stats.st_ctime,
                        "modified": stats.st_mtime,
                        "relative_path": item
                    })
            
            # Sort items: directories first, then files, both alphabetically
            items.sort(key=lambda x: (0 if x["type"] == "directory" else 1, x["name"].lower()))
            
            # Return results
            return {
                "success": True,
                "directory_path": directory_path,
                "pattern": pattern,
                "recursive": recursive,
                "items": items,
                "count": len(items)
            }
            
        except Exception as e:
            logger.error(f"Error listing directory {directory_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Error listing directory: {str(e)}",
                "exception_type": type(e).__name__
            }
