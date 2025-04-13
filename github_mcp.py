"""
GitHub MCP Server implementation for ME2AI.

This MCP server provides GitHub operations like repository access,
issues management, and code search capabilities.
"""
import os
import json
import logging
import asyncio
import base64
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import requests
from dotenv import load_dotenv

# Import MCP package
from mcp import MCPServer, register_tool, MCPToolInput

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("github-mcp")

class GitHubMCPServer(MCPServer):
    """MCP Server for GitHub operations."""

    def __init__(self):
        """Initialize the GitHub MCP server."""
        super().__init__("github")
        
        # GitHub API settings
        self.api_base_url = "https://api.github.com"
        self.api_key = None
        self._load_api_key()
        
        # Register tools
        self._register_tools()
        logger.info("✓ GitHub MCP Server initialized")

    def _load_api_key(self) -> None:
        """Load GitHub API key from environment variables."""
        # Load environment variables
        load_dotenv()
        
        # Get API key (try both potential variable names)
        self.api_key = os.getenv("GITHUB_API_KEY") or os.getenv("GITHUB_TOKEN")
        
        if not self.api_key:
            logger.warning("No GitHub API key found in environment variables")
            logger.warning("Please set GITHUB_API_KEY in your .env file")
            logger.warning("Limited functionality will be available")
        else:
            logger.info("✓ GitHub API key loaded successfully")

    def _register_tools(self) -> None:
        """Register all GitHub tools."""
        logger.info("Registering GitHub tools...")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for GitHub API requests."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ME2AI-GitHub-MCP-Server/1.0"
        }
        
        if self.api_key:
            headers["Authorization"] = f"token {self.api_key}"
            
        return headers

    @register_tool
    async def search_repositories(self, 
                                query: str, 
                                sort: str = "stars", 
                                order: str = "desc",
                                limit: int = 10) -> Dict[str, Any]:
        """Search for GitHub repositories.
        
        Args:
            query: Search query
            sort: Sort field (stars, forks, updated)
            order: Sort order (asc, desc)
            limit: Maximum number of results to return
            
        Returns:
            Dict containing success status and search results
        """
        logger.info(f"Searching repositories: {query}")
        
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
            
        except Exception as e:
            logger.error(f"Error searching repositories: {str(e)}")
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
        logger.info(f"Getting repository details: {repo_name}")
        
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
            
        except Exception as e:
            logger.error(f"Error getting repository details: {str(e)}")
            return {
                "success": False,
                "error": f"Error getting repository details: {str(e)}"
            }

    @register_tool
    async def list_repository_contents(self, 
                                     repo_name: str, 
                                     path: str = "", 
                                     ref: Optional[str] = None) -> Dict[str, Any]:
        """List contents of a GitHub repository.
        
        Args:
            repo_name: Repository name in format "owner/repo"
            path: Path within the repository (optional)
            ref: Git reference (branch, tag, commit) to use (optional)
            
        Returns:
            Dict containing success status and list of contents
        """
        logger.info(f"Listing repository contents: {repo_name} at path '{path}'")
        
        try:
            url = f"{self.api_base_url}/repos/{repo_name}/contents/{path}"
            params = {}
            if ref:
                params["ref"] = ref
                
            response = requests.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()
            
            contents = response.json()
            
            # Format results
            formatted_contents = []
            
            # Handle case when response is a list (directory) vs single item (file)
            if isinstance(contents, list):
                for item in contents:
                    formatted_contents.append({
                        "name": item.get("name"),
                        "path": item.get("path"),
                        "type": item.get("type"),  # "file" or "dir"
                        "size": item.get("size"),
                        "url": item.get("html_url"),
                        "download_url": item.get("download_url"),
                        "sha": item.get("sha")
                    })
            else:
                # Single file
                formatted_contents.append({
                    "name": contents.get("name"),
                    "path": contents.get("path"),
                    "type": contents.get("type"),
                    "size": contents.get("size"),
                    "url": contents.get("html_url"),
                    "download_url": contents.get("download_url"),
                    "sha": contents.get("sha")
                })
                
            return {
                "success": True,
                "repository": repo_name,
                "path": path,
                "ref": ref,
                "contents": formatted_contents
            }
            
        except Exception as e:
            logger.error(f"Error listing repository contents: {str(e)}")
            return {
                "success": False,
                "error": f"Error listing repository contents: {str(e)}"
            }

    @register_tool
    async def get_file_content(self, 
                            repo_name: str, 
                            file_path: str, 
                            ref: Optional[str] = None) -> Dict[str, Any]:
        """Get the content of a file from a GitHub repository.
        
        Args:
            repo_name: Repository name in format "owner/repo"
            file_path: Path to the file within the repository
            ref: Git reference (branch, tag, commit) to use (optional)
            
        Returns:
            Dict containing success status and file content
        """
        logger.info(f"Getting file content: {repo_name}/{file_path}")
        
        try:
            url = f"{self.api_base_url}/repos/{repo_name}/contents/{file_path}"
            params = {}
            if ref:
                params["ref"] = ref
                
            response = requests.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if it's a file
            if data.get("type") != "file":
                return {
                    "success": False,
                    "error": f"Path is not a file: {file_path}"
                }
                
            # Decode content
            content = base64.b64decode(data.get("content", "")).decode("utf-8")
            
            return {
                "success": True,
                "repository": repo_name,
                "path": file_path,
                "name": data.get("name"),
                "size": data.get("size"),
                "encoding": data.get("encoding"),
                "sha": data.get("sha"),
                "url": data.get("html_url"),
                "content": content
            }
            
        except Exception as e:
            logger.error(f"Error getting file content: {str(e)}")
            return {
                "success": False,
                "error": f"Error getting file content: {str(e)}"
            }

    @register_tool
    async def search_code(self, 
                        query: str, 
                        language: Optional[str] = None,
                        repo: Optional[str] = None,
                        limit: int = 10) -> Dict[str, Any]:
        """Search for code in GitHub repositories.
        
        Args:
            query: Search query
            language: Filter by programming language (optional)
            repo: Limit search to specific repo in format "owner/repo" (optional)
            limit: Maximum number of results to return
            
        Returns:
            Dict containing success status and search results
        """
        logger.info(f"Searching code: {query}")
        
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
            
        except Exception as e:
            logger.error(f"Error searching code: {str(e)}")
            return {
                "success": False,
                "error": f"Error searching code: {str(e)}"
            }

    @register_tool
    async def list_issues(self, 
                        repo_name: str, 
                        state: str = "open", 
                        sort: str = "created",
                        direction: str = "desc",
                        limit: int = 10) -> Dict[str, Any]:
        """List issues in a GitHub repository.
        
        Args:
            repo_name: Repository name in format "owner/repo"
            state: Filter by issue state (open, closed, all)
            sort: Sort field (created, updated, comments)
            direction: Sort direction (asc, desc)
            limit: Maximum number of issues to return
            
        Returns:
            Dict containing success status and list of issues
        """
        logger.info(f"Listing issues for {repo_name}")
        
        try:
            url = f"{self.api_base_url}/repos/{repo_name}/issues"
            params = {
                "state": state,
                "sort": sort,
                "direction": direction,
                "per_page": min(limit, 100)  # GitHub API limits to 100 per page
            }
            
            response = requests.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()
            
            issues = response.json()
            
            # Format results
            formatted_issues = []
            for issue in issues[:limit]:
                # Skip pull requests
                if "pull_request" in issue:
                    continue
                    
                formatted_issues.append({
                    "number": issue.get("number"),
                    "title": issue.get("title"),
                    "state": issue.get("state"),
                    "url": issue.get("html_url"),
                    "created_at": issue.get("created_at"),
                    "updated_at": issue.get("updated_at"),
                    "closed_at": issue.get("closed_at"),
                    "author": issue.get("user", {}).get("login"),
                    "author_url": issue.get("user", {}).get("html_url"),
                    "comments": issue.get("comments"),
                    "labels": [label.get("name") for label in issue.get("labels", [])]
                })
                
            return {
                "success": True,
                "repository": repo_name,
                "issues": formatted_issues,
                "returned_count": len(formatted_issues),
                "state_filter": state
            }
            
        except Exception as e:
            logger.error(f"Error listing issues: {str(e)}")
            return {
                "success": False,
                "error": f"Error listing issues: {str(e)}"
            }

    @register_tool
    async def get_issue_details(self, repo_name: str, issue_number: int) -> Dict[str, Any]:
        """Get detailed information about a specific GitHub issue.
        
        Args:
            repo_name: Repository name in format "owner/repo"
            issue_number: Issue number
            
        Returns:
            Dict containing success status and issue details
        """
        logger.info(f"Getting issue details: {repo_name}#{issue_number}")
        
        try:
            url = f"{self.api_base_url}/repos/{repo_name}/issues/{issue_number}"
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            
            issue = response.json()
            
            # Get comments
            comments_url = f"{self.api_base_url}/repos/{repo_name}/issues/{issue_number}/comments"
            comments_response = requests.get(comments_url, headers=self._get_headers())
            comments = comments_response.json() if comments_response.status_code == 200 else []
            
            # Format comments
            formatted_comments = []
            for comment in comments[:10]:  # Limit to first 10 comments
                formatted_comments.append({
                    "id": comment.get("id"),
                    "author": comment.get("user", {}).get("login"),
                    "author_url": comment.get("user", {}).get("html_url"),
                    "created_at": comment.get("created_at"),
                    "updated_at": comment.get("updated_at"),
                    "body": comment.get("body")
                })
                
            # Format result
            result = {
                "number": issue.get("number"),
                "title": issue.get("title"),
                "state": issue.get("state"),
                "url": issue.get("html_url"),
                "created_at": issue.get("created_at"),
                "updated_at": issue.get("updated_at"),
                "closed_at": issue.get("closed_at"),
                "author": issue.get("user", {}).get("login"),
                "author_url": issue.get("user", {}).get("html_url"),
                "body": issue.get("body"),
                "labels": [label.get("name") for label in issue.get("labels", [])],
                "assignees": [assignee.get("login") for assignee in issue.get("assignees", [])],
                "comments_count": issue.get("comments"),
                "comments": formatted_comments
            }
                
            return {
                "success": True,
                "repository": repo_name,
                "issue": result
            }
            
        except Exception as e:
            logger.error(f"Error getting issue details: {str(e)}")
            return {
                "success": False,
                "error": f"Error getting issue details: {str(e)}"
            }

# Start server if run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub MCP Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    server = GitHubMCPServer()
    asyncio.run(server.start())
