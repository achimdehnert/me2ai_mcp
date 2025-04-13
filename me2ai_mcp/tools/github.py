"""
GitHub-related tools for ME2AI MCP servers.

This module provides common tools for GitHub API operations
that can be used across different MCP servers.
"""
from typing import Dict, List, Any, Optional
import logging
import os
import base64
from dataclasses import dataclass
import requests

from ..base import BaseTool
from ..auth import AuthManager, TokenAuth

# Configure logging
logger = logging.getLogger("me2ai-mcp-tools-github")


@dataclass
class GitHubRepositoryTool(BaseTool):
    """Tool for GitHub repository operations."""
    
    name: str = "github_repository"
    description: str = "GitHub repository search and metadata"
    api_base_url: str = "https://api.github.com"
    
    def __post_init__(self):
        """Initialize with GitHub authentication if available."""
        super().__post_init__()
        self.auth = AuthManager.from_github_token()
        
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for GitHub API requests."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ME2AI-GitHub-MCP-Tools/1.0"
        }
        
        if self.auth.has_token():
            token = self.auth.get_token().token
            headers["Authorization"] = f"token {token}"
            
        return headers
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GitHub repository operations.
        
        Args:
            params: Dictionary containing:
                - operation: Operation to perform (search, get_details, list_contents)
                - query: Search query (for search operation)
                - repo_name: Repository name in format "owner/repo" (for get_details and list_contents)
                - path: Path within the repository (for list_contents)
                - ref: Git reference (branch, tag, commit) (for list_contents)
                - sort: Sort field for search results (stars, forks, updated)
                - order: Sort order (asc, desc)
                - limit: Maximum number of results to return
        
        Returns:
            Dictionary containing operation results
        """
        operation = params.get("operation")
        
        if not operation:
            return {
                "success": False,
                "error": "operation parameter is required"
            }
            
        if operation == "search":
            return await self._search_repositories(params)
        elif operation == "get_details":
            return await self._get_repository_details(params)
        elif operation == "list_contents":
            return await self._list_repository_contents(params)
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}"
            }
            
    async def _search_repositories(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search for GitHub repositories."""
        query = params.get("query")
        if not query:
            return {
                "success": False,
                "error": "query parameter is required for search operation"
            }
            
        sort = params.get("sort", "stars")
        order = params.get("order", "desc")
        limit = params.get("limit", 10)
        
        try:
            api_params = {
                "q": query,
                "sort": sort,
                "order": order,
                "per_page": min(limit, 100)  # GitHub API limits to 100 per page
            }
            
            url = f"{self.api_base_url}/search/repositories"
            response = requests.get(url, headers=self._get_headers(), params=api_params)
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
                        "url": repo.get("owner", {}).get("html_url"),
                        "avatar_url": repo.get("owner", {}).get("avatar_url")
                    }
                })
            
            return {
                "success": True,
                "query": query,
                "total_count": data.get("total_count", 0),
                "count": len(formatted_repos),
                "repositories": formatted_repos
            }
        except requests.RequestException as e:
            logger.error(f"Error searching repositories: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to search repositories: {str(e)}",
                "query": query
            }
        except Exception as e:
            logger.error(f"Unexpected error in search_repositories: {str(e)}")
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}",
                "query": query
            }
            
    async def _get_repository_details(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed information about a GitHub repository."""
        repo_name = params.get("repo_name")
        if not repo_name:
            return {
                "success": False,
                "error": "repo_name parameter is required for get_details operation"
            }
            
        try:
            url = f"{self.api_base_url}/repos/{repo_name}"
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            
            repo = response.json()
            
            # Get languages and topics
            languages_url = f"{self.api_base_url}/repos/{repo_name}/languages"
            languages_response = requests.get(languages_url, headers=self._get_headers())
            languages = languages_response.json() if languages_response.status_code == 200 else {}
            
            topics_url = f"{self.api_base_url}/repos/{repo_name}/topics"
            topics_headers = self._get_headers()
            topics_headers["Accept"] = "application/vnd.github.mercy-preview+json"
            topics_response = requests.get(topics_url, headers=topics_headers)
            topics = topics_response.json().get("names", []) if topics_response.status_code == 200 else []
            
            # Format detailed repository information
            repo_details = {
                "name": repo.get("name"),
                "full_name": repo.get("full_name"),
                "description": repo.get("description"),
                "url": repo.get("html_url"),
                "api_url": repo.get("url"),
                "created_at": repo.get("created_at"),
                "updated_at": repo.get("updated_at"),
                "pushed_at": repo.get("pushed_at"),
                "size": repo.get("size"),
                "stars": repo.get("stargazers_count"),
                "watchers": repo.get("watchers_count"),
                "forks": repo.get("forks_count"),
                "open_issues": repo.get("open_issues_count"),
                "default_branch": repo.get("default_branch"),
                "languages": languages,
                "topics": topics,
                "license": repo.get("license", {}).get("name") if repo.get("license") else None,
                "private": repo.get("private", False),
                "archived": repo.get("archived", False),
                "disabled": repo.get("disabled", False),
                "fork": repo.get("fork", False),
                "owner": {
                    "login": repo.get("owner", {}).get("login"),
                    "url": repo.get("owner", {}).get("html_url"),
                    "type": repo.get("owner", {}).get("type"),
                    "avatar_url": repo.get("owner", {}).get("avatar_url")
                }
            }
            
            return {
                "success": True,
                "repository": repo_details
            }
        except requests.RequestException as e:
            logger.error(f"Error getting repository details: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get repository details: {str(e)}",
                "repo_name": repo_name
            }
        except Exception as e:
            logger.error(f"Unexpected error in get_repository_details: {str(e)}")
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}",
                "repo_name": repo_name
            }
            
    async def _list_repository_contents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List contents of a GitHub repository."""
        repo_name = params.get("repo_name")
        if not repo_name:
            return {
                "success": False,
                "error": "repo_name parameter is required for list_contents operation"
            }
            
        path = params.get("path", "")
        ref = params.get("ref")
        
        try:
            url = f"{self.api_base_url}/repos/{repo_name}/contents/{path}"
            api_params = {}
            if ref:
                api_params["ref"] = ref
                
            response = requests.get(url, headers=self._get_headers(), params=api_params)
            response.raise_for_status()
            
            contents = response.json()
            formatted_contents = []
            
            # Handle single file response
            if not isinstance(contents, list):
                contents = [contents]
                
            for item in contents:
                formatted_contents.append({
                    "name": item.get("name"),
                    "path": item.get("path"),
                    "type": item.get("type"),
                    "size": item.get("size") if item.get("type") == "file" else None,
                    "download_url": item.get("download_url"),
                    "html_url": item.get("html_url"),
                    "git_url": item.get("git_url")
                })
            
            # Sort directories first, then files
            formatted_contents.sort(key=lambda x: (0 if x["type"] == "dir" else 1, x["name"]))
            
            return {
                "success": True,
                "repo_name": repo_name,
                "path": path,
                "ref": ref,
                "contents": formatted_contents
            }
        except requests.RequestException as e:
            logger.error(f"Error listing repository contents: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to list repository contents: {str(e)}",
                "repo_name": repo_name,
                "path": path
            }
        except Exception as e:
            logger.error(f"Unexpected error in list_repository_contents: {str(e)}")
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}",
                "repo_name": repo_name,
                "path": path
            }


@dataclass
class GitHubCodeTool(BaseTool):
    """Tool for GitHub code operations."""
    
    name: str = "github_code"
    description: str = "GitHub code search and file operations"
    api_base_url: str = "https://api.github.com"
    
    def __post_init__(self):
        """Initialize with GitHub authentication if available."""
        super().__post_init__()
        self.auth = AuthManager.from_github_token()
        
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for GitHub API requests."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ME2AI-GitHub-MCP-Tools/1.0"
        }
        
        if self.auth.has_token():
            token = self.auth.get_token().token
            headers["Authorization"] = f"token {token}"
            
        return headers
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GitHub code operations.
        
        Args:
            params: Dictionary containing:
                - operation: Operation to perform (search, get_file)
                - query: Search query (for search operation)
                - language: Filter by programming language (for search operation)
                - repo: Limit search to specific repo in format "owner/repo" (for search operation)
                - repo_name: Repository name in format "owner/repo" (for get_file)
                - file_path: Path to the file within the repository (for get_file)
                - ref: Git reference (branch, tag, commit) (for get_file)
                - limit: Maximum number of results to return
        
        Returns:
            Dictionary containing operation results
        """
        operation = params.get("operation")
        
        if not operation:
            return {
                "success": False,
                "error": "operation parameter is required"
            }
            
        if operation == "search":
            return await self._search_code(params)
        elif operation == "get_file":
            return await self._get_file_content(params)
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}"
            }
            
    async def _search_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search for code in GitHub repositories."""
        query = params.get("query")
        if not query:
            return {
                "success": False,
                "error": "query parameter is required for search operation"
            }
            
        language = params.get("language")
        repo = params.get("repo")
        limit = params.get("limit", 10)
        
        try:
            # Build the search query
            search_query = query
            if language:
                search_query += f" language:{language}"
            if repo:
                search_query += f" repo:{repo}"
                
            api_params = {
                "q": search_query,
                "per_page": min(limit, 100)  # GitHub API limits to 100 per page
            }
            
            url = f"{self.api_base_url}/search/code"
            headers = self._get_headers()
            
            # Code search requires specific accept header
            headers["Accept"] = "application/vnd.github.v3.text-match+json"
            
            response = requests.get(url, headers=headers, params=api_params)
            response.raise_for_status()
            
            data = response.json()
            items = data.get("items", [])
            
            # Format results
            formatted_results = []
            for item in items[:limit]:
                # Extract matches if available
                matches = []
                for match in item.get("text_matches", []):
                    matches.append({
                        "fragment": match.get("fragment", ""),
                        "property": match.get("property", "")
                    })
                
                formatted_results.append({
                    "name": item.get("name", ""),
                    "path": item.get("path", ""),
                    "repository": {
                        "name": item.get("repository", {}).get("name", ""),
                        "full_name": item.get("repository", {}).get("full_name", ""),
                        "url": item.get("repository", {}).get("html_url", "")
                    },
                    "html_url": item.get("html_url", ""),
                    "git_url": item.get("git_url", ""),
                    "matches": matches
                })
            
            return {
                "success": True,
                "query": query,
                "language": language,
                "repo": repo,
                "total_count": data.get("total_count", 0),
                "count": len(formatted_results),
                "results": formatted_results
            }
        except requests.RequestException as e:
            logger.error(f"Error searching code: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to search code: {str(e)}",
                "query": query
            }
        except Exception as e:
            logger.error(f"Unexpected error in search_code: {str(e)}")
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}",
                "query": query
            }
            
    async def _get_file_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get the content of a file from a GitHub repository."""
        repo_name = params.get("repo_name")
        file_path = params.get("file_path")
        
        if not repo_name:
            return {
                "success": False,
                "error": "repo_name parameter is required for get_file operation"
            }
            
        if not file_path:
            return {
                "success": False,
                "error": "file_path parameter is required for get_file operation"
            }
            
        ref = params.get("ref")
        
        try:
            url = f"{self.api_base_url}/repos/{repo_name}/contents/{file_path}"
            api_params = {}
            if ref:
                api_params["ref"] = ref
                
            response = requests.get(url, headers=self._get_headers(), params=api_params)
            response.raise_for_status()
            
            file_data = response.json()
            
            if file_data.get("type") != "file":
                return {
                    "success": False,
                    "error": "Specified path is not a file",
                    "repo_name": repo_name,
                    "file_path": file_path
                }
            
            # Decode content from base64
            encoded_content = file_data.get("content", "")
            # GitHub API returns base64 with newlines, remove them first
            encoded_content = encoded_content.replace("\n", "")
            content = base64.b64decode(encoded_content).decode("utf-8")
            
            return {
                "success": True,
                "repo_name": repo_name,
                "file_path": file_path,
                "ref": ref,
                "size": file_data.get("size", 0),
                "name": file_data.get("name", ""),
                "content": content,
                "html_url": file_data.get("html_url"),
                "download_url": file_data.get("download_url"),
                "encoding": "utf-8"  # We're decoding as UTF-8
            }
        except UnicodeDecodeError:
            # Handle binary files
            logger.warning(f"Binary file detected: {file_path}")
            return {
                "success": False,
                "error": "Binary file detected, content not returned",
                "repo_name": repo_name,
                "file_path": file_path,
                "is_binary": True
            }
        except requests.RequestException as e:
            logger.error(f"Error getting file content: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get file content: {str(e)}",
                "repo_name": repo_name,
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Unexpected error in get_file_content: {str(e)}")
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}",
                "repo_name": repo_name,
                "file_path": file_path
            }


@dataclass
class GitHubIssuesTool(BaseTool):
    """Tool for GitHub issues operations."""
    
    name: str = "github_issues"
    description: str = "GitHub issues management"
    api_base_url: str = "https://api.github.com"
    
    def __post_init__(self):
        """Initialize with GitHub authentication if available."""
        super().__post_init__()
        self.auth = AuthManager.from_github_token()
        
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for GitHub API requests."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ME2AI-GitHub-MCP-Tools/1.0"
        }
        
        if self.auth.has_token():
            token = self.auth.get_token().token
            headers["Authorization"] = f"token {token}"
            
        return headers
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GitHub issues operations.
        
        Args:
            params: Dictionary containing:
                - operation: Operation to perform (list, get_details)
                - repo_name: Repository name in format "owner/repo"
                - issue_number: Issue number (for get_details)
                - state: Filter by issue state (open, closed, all) (for list)
                - sort: Sort field (created, updated, comments) (for list)
                - direction: Sort direction (asc, desc) (for list)
                - limit: Maximum number of results to return
        
        Returns:
            Dictionary containing operation results
        """
        operation = params.get("operation")
        
        if not operation:
            return {
                "success": False,
                "error": "operation parameter is required"
            }
            
        if operation == "list":
            return await self._list_issues(params)
        elif operation == "get_details":
            return await self._get_issue_details(params)
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}"
            }
            
    async def _list_issues(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List issues in a GitHub repository."""
        repo_name = params.get("repo_name")
        if not repo_name:
            return {
                "success": False,
                "error": "repo_name parameter is required for list operation"
            }
            
        state = params.get("state", "open")
        sort = params.get("sort", "created")
        direction = params.get("direction", "desc")
        limit = params.get("limit", 10)
        
        # Validate state parameter
        if state not in ["open", "closed", "all"]:
            return {
                "success": False,
                "error": "Invalid state parameter. Must be one of: open, closed, all",
                "repo_name": repo_name
            }
            
        # Validate sort parameter
        if sort not in ["created", "updated", "comments"]:
            return {
                "success": False,
                "error": "Invalid sort parameter. Must be one of: created, updated, comments",
                "repo_name": repo_name
            }
            
        # Validate direction parameter
        if direction not in ["asc", "desc"]:
            return {
                "success": False,
                "error": "Invalid direction parameter. Must be one of: asc, desc",
                "repo_name": repo_name
            }
        
        try:
            url = f"{self.api_base_url}/repos/{repo_name}/issues"
            api_params = {
                "state": state,
                "sort": sort,
                "direction": direction,
                "per_page": min(limit, 100)
            }
            
            response = requests.get(url, headers=self._get_headers(), params=api_params)
            response.raise_for_status()
            
            issues = response.json()
            
            # Format results
            formatted_issues = []
            for issue in issues[:limit]:
                # Skip pull requests (they show up in the issues endpoint)
                if "pull_request" in issue:
                    continue
                    
                labels = [label.get("name") for label in issue.get("labels", [])]
                
                formatted_issues.append({
                    "number": issue.get("number"),
                    "title": issue.get("title"),
                    "state": issue.get("state"),
                    "url": issue.get("html_url"),
                    "created_at": issue.get("created_at"),
                    "updated_at": issue.get("updated_at"),
                    "closed_at": issue.get("closed_at"),
                    "user": {
                        "login": issue.get("user", {}).get("login"),
                        "url": issue.get("user", {}).get("html_url")
                    },
                    "labels": labels,
                    "comments": issue.get("comments", 0),
                    "body_preview": issue.get("body", "")[:200] + ("..." if issue.get("body", "") and len(issue.get("body", "")) > 200 else "")
                })
            
            return {
                "success": True,
                "repo_name": repo_name,
                "state": state,
                "count": len(formatted_issues),
                "issues": formatted_issues
            }
        except requests.RequestException as e:
            logger.error(f"Error listing issues: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to list issues: {str(e)}",
                "repo_name": repo_name
            }
        except Exception as e:
            logger.error(f"Unexpected error in list_issues: {str(e)}")
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}",
                "repo_name": repo_name
            }
            
    async def _get_issue_details(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed information about a specific GitHub issue."""
        repo_name = params.get("repo_name")
        issue_number = params.get("issue_number")
        
        if not repo_name:
            return {
                "success": False,
                "error": "repo_name parameter is required for get_details operation"
            }
            
        if not issue_number:
            return {
                "success": False,
                "error": "issue_number parameter is required for get_details operation"
            }
        
        try:
            url = f"{self.api_base_url}/repos/{repo_name}/issues/{issue_number}"
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            
            issue = response.json()
            
            # Check if this is a pull request
            if "pull_request" in issue:
                return {
                    "success": False,
                    "error": "The specified issue is a pull request, not an issue",
                    "repo_name": repo_name,
                    "issue_number": issue_number,
                    "pull_request_url": issue.get("pull_request", {}).get("html_url")
                }
            
            # Get comments
            comments_url = f"{self.api_base_url}/repos/{repo_name}/issues/{issue_number}/comments"
            comments_response = requests.get(comments_url, headers=self._get_headers())
            comments = []
            
            if comments_response.status_code == 200:
                comments_data = comments_response.json()
                for comment in comments_data:
                    comments.append({
                        "user": {
                            "login": comment.get("user", {}).get("login"),
                            "url": comment.get("user", {}).get("html_url")
                        },
                        "created_at": comment.get("created_at"),
                        "updated_at": comment.get("updated_at"),
                        "body": comment.get("body")
                    })
            
            # Format detailed issue information
            labels = [label.get("name") for label in issue.get("labels", [])]
            
            issue_details = {
                "number": issue.get("number"),
                "title": issue.get("title"),
                "state": issue.get("state"),
                "url": issue.get("html_url"),
                "created_at": issue.get("created_at"),
                "updated_at": issue.get("updated_at"),
                "closed_at": issue.get("closed_at"),
                "user": {
                    "login": issue.get("user", {}).get("login"),
                    "url": issue.get("user", {}).get("html_url"),
                    "avatar_url": issue.get("user", {}).get("avatar_url")
                },
                "labels": labels,
                "assignees": [
                    {
                        "login": assignee.get("login"),
                        "url": assignee.get("html_url")
                    }
                    for assignee in issue.get("assignees", [])
                ],
                "comments_count": issue.get("comments", 0),
                "comments": comments,
                "body": issue.get("body", "")
            }
            
            return {
                "success": True,
                "repo_name": repo_name,
                "issue_number": issue_number,
                "issue": issue_details
            }
        except requests.RequestException as e:
            logger.error(f"Error getting issue details: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get issue details: {str(e)}",
                "repo_name": repo_name,
                "issue_number": issue_number
            }
        except Exception as e:
            logger.error(f"Unexpected error in get_issue_details: {str(e)}")
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}",
                "repo_name": repo_name,
                "issue_number": issue_number
            }
