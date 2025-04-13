"""
GitHub MCP Server implementation for ME2AI using the enhanced ME2AI MCP framework.

This MCP server provides GitHub operations like repository access,
issues management, and code search capabilities with improved error handling,
logging, and authentication via the ME2AI MCP framework.
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

# Import ME2AI MCP framework
from me2ai_mcp import ME2AIMCPServer, register_tool
from me2ai_mcp.auth import AuthManager, TokenAuth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("github-mcp")


class GitHubMCPServer(ME2AIMCPServer):
    """Enhanced MCP Server for GitHub operations using ME2AI MCP framework."""

    def __init__(self):
        """Initialize the GitHub MCP server with the ME2AI MCP framework."""
        super().__init__(
            server_name="github-mcp",
            description="GitHub operations service for repository access, code search, and issues management",
            version="1.0.0"
        )
        
        # GitHub API settings
        self.api_base_url = "https://api.github.com"
        
        # Set up authentication using ME2AI MCP auth module
        self.auth = AuthManager.from_github_token()
        if not self.auth.has_token():
            logger.warning("No GitHub API key found in environment variables")
            logger.warning("Please set GITHUB_API_KEY or GITHUB_TOKEN in your .env file")
            logger.warning("Limited functionality will be available")
        else:
            logger.info("✓ GitHub API key loaded successfully")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for GitHub API requests."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ME2AI-GitHub-MCP-Server/1.0"
        }
        
        if self.auth.has_token():
            token = self.auth.get_token().token
            headers["Authorization"] = f"token {token}"
            
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
        logger.info(f"Listing repository contents: {repo_name}, path: {path}, ref: {ref}")
        
        try:
            url = f"{self.api_base_url}/repos/{repo_name}/contents/{path}"
            params = {}
            if ref:
                params["ref"] = ref
                
            response = requests.get(url, headers=self._get_headers(), params=params)
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
        logger.info(f"Getting file content: {repo_name}, path: {file_path}, ref: {ref}")
        
        try:
            url = f"{self.api_base_url}/repos/{repo_name}/contents/{file_path}"
            params = {}
            if ref:
                params["ref"] = ref
                
            response = requests.get(url, headers=self._get_headers(), params=params)
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
        logger.info(f"Searching code: {query}, language: {language}, repo: {repo}")
        
        try:
            # Build the search query
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
            headers = self._get_headers()
            
            # Code search requires specific accept header
            headers["Accept"] = "application/vnd.github.v3.text-match+json"
            
            response = requests.get(url, headers=headers, params=params)
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
        logger.info(f"Listing issues: {repo_name}, state: {state}")
        
        try:
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
            
            url = f"{self.api_base_url}/repos/{repo_name}/issues"
            params = {
                "state": state,
                "sort": sort,
                "direction": direction,
                "per_page": min(limit, 100)
            }
            
            response = requests.get(url, headers=self._get_headers(), params=params)
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

    @register_tool
    async def get_issue_details(self, repo_name: str, issue_number: int) -> Dict[str, Any]:
        """Get detailed information about a specific GitHub issue.
        
        Args:
            repo_name: Repository name in format "owner/repo"
            issue_number: Issue number
            
        Returns:
            Dict containing success status and issue details
        """
        logger.info(f"Getting issue details: {repo_name}, issue: {issue_number}")
        
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


async def main():
    """Run the GitHub MCP server."""
    server = GitHubMCPServer()
    await server.start()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub MCP Server")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
