"""
GitHub MCP Server implementation for ME2AI using the ME2AI MCP framework.

This MCP server provides GitHub operations like repository access,
issues management, and code search capabilities with improved error handling
and standardized response formats.
"""
import os
import logging
import asyncio
import argparse
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass

# Import ME2AI MCP package
from me2ai.mcp import ME2AIMCPServer, register_tool
from me2ai.mcp.auth import AuthManager, TokenAuth
from me2ai.mcp.utils import sanitize_input, format_response

# Try importing optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("Requests library not available, GitHub MCP server will not function properly")


class GitHubMCPServer(ME2AIMCPServer):
    """MCP Server for GitHub operations using ME2AI MCP framework."""

    def __init__(self, debug: bool = False):
        """Initialize the GitHub MCP server.
        
        Args:
            debug: Whether to enable debug logging
        """
        super().__init__(
            server_name="github",
            description="GitHub API integration for ME2AI agents",
            version="0.1.0",
            debug=debug
        )
        
        # Set up authentication
        self.auth = AuthManager.from_github_token()
        self.api_base_url = "https://api.github.com"
        
        # Register tools
        self._register_tools()
        
        if not self.auth.providers:
            self.logger.warning("No GitHub token found in environment variables")
            self.logger.warning("Limited functionality will be available")
        else:
            self.logger.info("✓ GitHub MCP Server initialized with authentication")

    def _register_tools(self):
        """Register GitHub tools with the MCP server."""
        self.logger.info("Registering GitHub API tools...")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for GitHub API requests with authentication.
        
        Returns:
            Dictionary of HTTP headers for API requests
        """
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ME2AI-GitHub-MCP-Server/1.0"
        }
        
        # Add authentication if available
        auth_headers = self.auth.get_auth_headers()
        headers.update(auth_headers)
            
        return headers

    @register_tool
    async def search_repositories(
        self, 
        query: str, 
        sort: str = "stars", 
        order: str = "desc",
        limit: int = 10
    ) -> Dict[str, Any]:
        """Search for GitHub repositories.
        
        Args:
            query: Search query
            sort: Sort field (stars, forks, updated)
            order: Sort order (asc, desc)
            limit: Maximum number of results to return
            
        Returns:
            Dict containing success status and search results
        """
        if not REQUESTS_AVAILABLE:
            return {
                "success": False,
                "error": "Requests library not available, cannot perform GitHub API requests"
            }
        
        # Validate and sanitize inputs
        query = sanitize_input(query)
        sort = sanitize_input(sort)
        order = sanitize_input(order)
        
        # Validate sort parameter
        valid_sort_options = ["stars", "forks", "updated", "help-wanted-issues"]
        if sort not in valid_sort_options:
            return {
                "success": False,
                "error": f"Invalid sort parameter: {sort}. Valid options are: {', '.join(valid_sort_options)}"
            }
            
        # Validate order parameter
        valid_order_options = ["asc", "desc"]
        if order not in valid_order_options:
            return {
                "success": False,
                "error": f"Invalid order parameter: {order}. Valid options are: {', '.join(valid_order_options)}"
            }
        
        try:
            params = {
                "q": query,
                "sort": sort,
                "order": order,
                "per_page": min(limit, 100)  # GitHub API limits to 100 per page
            }
            
            url = f"{self.api_base_url}/search/repositories"
            response = requests.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()
            
            data = response.json()
            repositories = data.get("items", [])
            
            # Format results
            formatted_repos = []
            for repo in repositories[:limit]:
                formatted_repos.append({
                    "name": repo.get("name"),
                    "full_name": repo.get("full_name"),
                    "url": repo.get("html_url"),
                    "description": repo.get("description"),
                    "stars": repo.get("stargazers_count"),
                    "forks": repo.get("forks_count"),
                    "language": repo.get("language"),
                    "created_at": repo.get("created_at"),
                    "updated_at": repo.get("updated_at"),
                    "owner": {
                        "login": repo.get("owner", {}).get("login"),
                        "avatar_url": repo.get("owner", {}).get("avatar_url"),
                        "url": repo.get("owner", {}).get("html_url")
                    }
                })
                
            return {
                "success": True,
                "query": query,
                "repositories": formatted_repos,
                "total_count": data.get("total_count", 0),
                "returned_count": len(formatted_repos)
            }
            
        except requests.RequestException as e:
            self.logger.error(f"GitHub API request error: {str(e)}")
            return {
                "success": False,
                "error": f"GitHub API request error: {str(e)}",
                "status_code": getattr(e.response, "status_code", None) if hasattr(e, "response") else None
            }
        except Exception as e:
            self.logger.error(f"Error searching repositories: {str(e)}")
            return {
                "success": False,
                "error": f"Error searching repositories: {str(e)}"
            }

    @register_tool
    async def get_repository_details(self, repo_name: str) -> Dict[str, Any]:
        """Get detailed information about a GitHub repository.
        
        Args:
            repo_name: Repository name in format "owner/repo"
            
        Returns:
            Dict containing success status and repository details
        """
        if not REQUESTS_AVAILABLE:
            return {
                "success": False,
                "error": "Requests library not available, cannot perform GitHub API requests"
            }
        
        # Validate and sanitize input
        repo_name = sanitize_input(repo_name)
        
        # Make sure repo_name is in the correct format
        if not "/" in repo_name:
            return {
                "success": False,
                "error": f"Invalid repository name format: {repo_name}. Expected format: owner/repo"
            }
        
        try:
            url = f"{self.api_base_url}/repos/{repo_name}"
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            
            repo = response.json()
            
            # Get languages
            languages_url = repo.get("languages_url")
            languages_response = requests.get(languages_url, headers=self._get_headers())
            languages = languages_response.json() if languages_response.status_code == 200 else {}
            
            # Format results
            result = {
                "name": repo.get("name"),
                "full_name": repo.get("full_name"),
                "description": repo.get("description"),
                "url": repo.get("html_url"),
                "api_url": repo.get("url"),
                "default_branch": repo.get("default_branch"),
                "created_at": repo.get("created_at"),
                "updated_at": repo.get("updated_at"),
                "pushed_at": repo.get("pushed_at"),
                "size": repo.get("size"),
                "stars": repo.get("stargazers_count"),
                "watchers": repo.get("watchers_count"),
                "forks": repo.get("forks_count"),
                "open_issues": repo.get("open_issues_count"),
                "license": repo.get("license", {}).get("name"),
                "topics": repo.get("topics", []),
                "languages": languages,
                "has_wiki": repo.get("has_wiki"),
                "has_pages": repo.get("has_pages"),
                "has_projects": repo.get("has_projects"),
                "has_discussions": repo.get("has_discussions"),
                "owner": {
                    "login": repo.get("owner", {}).get("login"),
                    "avatar_url": repo.get("owner", {}).get("avatar_url"),
                    "url": repo.get("owner", {}).get("html_url")
                }
            }
                
            return {
                "success": True,
                "repository": result
            }
            
        except requests.RequestException as e:
            self.logger.error(f"GitHub API request error: {str(e)}")
            return {
                "success": False,
                "error": f"GitHub API request error: {str(e)}",
                "status_code": getattr(e.response, "status_code", None) if hasattr(e, "response") else None
            }
        except Exception as e:
            self.logger.error(f"Error getting repository details: {str(e)}")
            return {
                "success": False,
                "error": f"Error getting repository details: {str(e)}"
            }

    @register_tool
    async def search_code(
        self, 
        query: str, 
        language: Optional[str] = None,
        repo: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Search for code in GitHub repositories.
        
        Args:
            query: Search query
            language: Filter by programming language (optional)
            repo: Limit search to specific repo in format "owner/repo" (optional)
            limit: Maximum number of results to return
            
        Returns:
            Dict containing success status and search results
        """
        if not REQUESTS_AVAILABLE:
            return {
                "success": False,
                "error": "Requests library not available, cannot perform GitHub API requests"
            }
        
        # Validate and sanitize inputs
        query = sanitize_input(query)
        if language:
            language = sanitize_input(language)
        if repo:
            repo = sanitize_input(repo)
        
        try:
            # Build query string
            search_query = query
            if language:
                search_query += f" language:{language}"
            if repo:
                search_query += f" repo:{repo}"
                
            params = {
                "q": search_query,
                "per_page": min(limit, 100)  # GitHub API limits to 100 per page
            }
            
            url = f"{self.api_base_url}/search/code"
            response = requests.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()
            
            data = response.json()
            items = data.get("items", [])
            
            # Format results
            formatted_items = []
            for item in items[:limit]:
                # Get file details
                repository = item.get("repository", {})
                
                formatted_items.append({
                    "repository": {
                        "name": repository.get("name"),
                        "full_name": repository.get("full_name"),
                        "url": repository.get("html_url")
                    },
                    "name": item.get("name"),
                    "path": item.get("path"),
                    "url": item.get("html_url"),
                    "sha": item.get("sha"),
                    "score": item.get("score")
                })
                
            return {
                "success": True,
                "query": query,
                "language": language,
                "repository": repo,
                "results": formatted_items,
                "total_count": data.get("total_count", 0),
                "returned_count": len(formatted_items)
            }
            
        except requests.RequestException as e:
            self.logger.error(f"GitHub API request error: {str(e)}")
            return {
                "success": False,
                "error": f"GitHub API request error: {str(e)}",
                "status_code": getattr(e.response, "status_code", None) if hasattr(e, "response") else None
            }
        except Exception as e:
            self.logger.error(f"Error searching code: {str(e)}")
            return {
                "success": False,
                "error": f"Error searching code: {str(e)}"
            }


def main():
    """Run the GitHub MCP server."""
    parser = argparse.ArgumentParser(description="GitHub MCP Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Create and start server
    server = GitHubMCPServer(debug=args.debug)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error(f"Server error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
