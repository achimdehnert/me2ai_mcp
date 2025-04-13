"""
Filesystem MCP Server implementation for ME2AI.

This MCP server provides filesystem operations like reading, writing, and listing files.
"""
import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

# Import MCP package
from mcp import MCPServer, register_tool, MCPToolInput

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("filesystem-mcp")

class FilesystemMCPServer(MCPServer):
    """MCP Server for filesystem operations."""

    def __init__(self, max_file_size: int = 1024 * 1024 * 5):
        """Initialize the Filesystem MCP server.
        
        Args:
            max_file_size: Maximum file size in bytes (default: 5MB)
        """
        super().__init__("filesystem")
        self.max_file_size = max_file_size
        
        # Register tools
        self._register_tools()
        logger.info("✓ Filesystem MCP Server initialized")

    def _register_tools(self) -> None:
        """Register all filesystem tools."""
        logger.info("Registering filesystem tools...")

    @register_tool
    async def read_file(self, file_path: str) -> Dict[str, Any]:
        """Read the contents of a file.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            Dict containing success status, file content, and metadata
        """
        logger.info(f"Reading file: {file_path}")
        
        try:
            # Normalize and validate path
            file_path = os.path.abspath(file_path)
            
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
                
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
                    "error": f"File too large ({file_size} bytes). Maximum size is {self.max_file_size} bytes."
                }
                
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Get file metadata
            stats = os.stat(file_path)
            
            return {
                "success": True,
                "content": content,
                "metadata": {
                    "path": file_path,
                    "size": file_size,
                    "created": stats.st_ctime,
                    "modified": stats.st_mtime,
                    "extension": os.path.splitext(file_path)[1],
                    "filename": os.path.basename(file_path)
                }
            }
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Error reading file: {str(e)}"
            }

    @register_tool
    async def write_file(self, file_path: str, content: str, overwrite: bool = False) -> Dict[str, Any]:
        """Write content to a file.
        
        Args:
            file_path: Path to the file to write
            content: Content to write to the file
            overwrite: Whether to overwrite existing files
            
        Returns:
            Dict containing success status and metadata
        """
        logger.info(f"Writing to file: {file_path} (overwrite={overwrite})")
        
        try:
            # Normalize and validate path
            file_path = os.path.abspath(file_path)
            
            # Check if file exists and overwrite setting
            if os.path.exists(file_path) and not overwrite:
                return {
                    "success": False,
                    "error": f"File already exists and overwrite=False: {file_path}"
                }
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            # Get file metadata
            file_size = os.path.getsize(file_path)
            stats = os.stat(file_path)
            
            return {
                "success": True,
                "metadata": {
                    "path": file_path,
                    "size": file_size,
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
                "error": f"Error writing to file: {str(e)}"
            }

    @register_tool
    async def list_directory(self, directory_path: str, pattern: Optional[str] = None) -> Dict[str, Any]:
        """List contents of a directory.
        
        Args:
            directory_path: Path to the directory
            pattern: Optional glob pattern to filter results
            
        Returns:
            Dict containing success status and list of files/directories
        """
        logger.info(f"Listing directory: {directory_path} (pattern={pattern})")
        
        try:
            # Normalize and validate path
            directory_path = os.path.abspath(directory_path)
            
            if not os.path.exists(directory_path):
                return {
                    "success": False,
                    "error": f"Directory not found: {directory_path}"
                }
                
            if not os.path.isdir(directory_path):
                return {
                    "success": False,
                    "error": f"Path is not a directory: {directory_path}"
                }
                
            # List directory contents
            items = []
            path_obj = Path(directory_path)
            
            if pattern:
                entries = list(path_obj.glob(pattern))
            else:
                entries = list(path_obj.iterdir())
                
            for entry in entries:
                item_type = "directory" if entry.is_dir() else "file"
                stats = entry.stat()
                
                items.append({
                    "name": entry.name,
                    "path": str(entry),
                    "type": item_type,
                    "size": stats.st_size if item_type == "file" else None,
                    "created": stats.st_ctime,
                    "modified": stats.st_mtime,
                    "extension": entry.suffix if item_type == "file" else None
                })
                
            return {
                "success": True,
                "directory": directory_path,
                "items": items,
                "item_count": len(items)
            }
            
        except Exception as e:
            logger.error(f"Error listing directory {directory_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Error listing directory: {str(e)}"
            }

    @register_tool
    async def search_files(self, 
                          directory_path: str, 
                          query: str, 
                          file_extensions: Optional[List[str]] = None,
                          recursive: bool = True,
                          max_results: int = 50) -> Dict[str, Any]:
        """Search for files containing specific text.
        
        Args:
            directory_path: Base directory to search in
            query: Text to search for
            file_extensions: List of file extensions to include (e.g. ['.py', '.txt'])
            recursive: Whether to search recursively in subdirectories
            max_results: Maximum number of results to return
            
        Returns:
            Dict containing success status and list of matching files
        """
        logger.info(f"Searching for '{query}' in {directory_path}")
        
        try:
            # Normalize and validate path
            directory_path = os.path.abspath(directory_path)
            
            if not os.path.exists(directory_path):
                return {
                    "success": False,
                    "error": f"Directory not found: {directory_path}"
                }
                
            if not os.path.isdir(directory_path):
                return {
                    "success": False,
                    "error": f"Path is not a directory: {directory_path}"
                }
                
            # Convert query to lowercase for case-insensitive search
            query = query.lower()
            
            # Search for files
            matches = []
            
            for root, _, files in os.walk(directory_path):
                if not recursive and root != directory_path:
                    continue
                    
                for filename in files:
                    # Check file extension if specified
                    if file_extensions:
                        file_ext = os.path.splitext(filename)[1]
                        if file_ext not in file_extensions:
                            continue
                            
                    file_path = os.path.join(root, filename)
                    
                    # Skip files that are too large
                    if os.path.getsize(file_path) > self.max_file_size:
                        continue
                        
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().lower()
                            
                        if query in content:
                            # Find line numbers where query appears
                            lines = content.split('\n')
                            matching_lines = [
                                (i+1, line) 
                                for i, line in enumerate(lines) 
                                if query in line.lower()
                            ]
                            
                            # Only keep a few matching lines
                            if len(matching_lines) > 5:
                                matching_lines = matching_lines[:5]
                                
                            matches.append({
                                "path": file_path,
                                "filename": filename,
                                "relative_path": os.path.relpath(file_path, directory_path),
                                "matching_lines": [
                                    {"line_number": line_num, "content": line[:100] + ("..." if len(line) > 100 else "")}
                                    for line_num, line in matching_lines
                                ],
                                "match_count": len(matching_lines)
                            })
                            
                            if len(matches) >= max_results:
                                break
                    except (UnicodeDecodeError, IOError):
                        # Skip binary files or files with encoding issues
                        continue
                        
                if len(matches) >= max_results:
                    break
                    
            return {
                "success": True,
                "query": query,
                "matches": matches,
                "match_count": len(matches),
                "max_reached": len(matches) >= max_results
            }
            
        except Exception as e:
            logger.error(f"Error searching in {directory_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Error searching: {str(e)}"
            }

    @register_tool
    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get detailed information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict containing success status and file metadata
        """
        logger.info(f"Getting file info: {file_path}")
        
        try:
            # Normalize and validate path
            file_path = os.path.abspath(file_path)
            
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"File/directory not found: {file_path}"
                }
                
            # Get stats
            stats = os.stat(file_path)
            is_dir = os.path.isdir(file_path)
            
            # Build response
            info = {
                "path": file_path,
                "name": os.path.basename(file_path),
                "type": "directory" if is_dir else "file",
                "size": stats.st_size,
                "created": stats.st_ctime,
                "modified": stats.st_mtime,
                "accessed": stats.st_atime,
                "permissions": stats.st_mode,
            }
            
            # Add file-specific info
            if not is_dir:
                info["extension"] = os.path.splitext(file_path)[1]
                
                # Try to detect if it's a text file
                try:
                    is_text = False
                    with open(file_path, 'rb') as f:
                        sample = f.read(1024)
                        is_text = b'\0' not in sample  # Crude heuristic
                    info["is_text"] = is_text
                except:
                    info["is_text"] = False
            
            return {
                "success": True,
                "info": info
            }
            
        except Exception as e:
            logger.error(f"Error getting info for {file_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Error getting file info: {str(e)}"
            }

# Start server if run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filesystem MCP Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--max-file-size", type=int, default=5*1024*1024, 
                       help="Maximum file size in bytes (default: 5MB)")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    server = FilesystemMCPServer(max_file_size=args.max_file_size)
    asyncio.run(server.start())
