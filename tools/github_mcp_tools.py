"""
GitHub MCP tools for ME2AI agents.

This module provides tool interfaces for the GitHub MCP server.
"""
from typing import Dict, List, Any, Optional, Union
import json
import requests
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

class SearchRepositoriesTool(BaseTool):
    """Tool for searching GitHub repositories using the GitHub MCP server."""
    
    name = "search_github_repositories"
    description = """Search for GitHub repositories based on keywords, topics, or criteria.
    Useful for finding open-source projects, libraries, or references.
    Input should be a string with search terms or a JSON string with 'query', and optional 'sort', 'order', and 'limit' fields."""
    
    def _run(self, tool_input: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run the tool to search repositories."""
        try:
            # Parse input - handle both string and JSON
            sort = "stars"
            order = "desc"
            limit = 10
            
            if tool_input.startswith("{") and tool_input.endswith("}"):
                try:
                    input_data = json.loads(tool_input)
                    query = input_data.get("query")
                    sort = input_data.get("sort", sort)
                    order = input_data.get("order", order)
                    limit = input_data.get("limit", limit)
                except:
                    query = tool_input
            else:
                query = tool_input
                
            if not query:
                return "Error: Search query must be provided"
                
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers={"Content-Type": "application/json"},
                json={
                    "serverName": "github",
                    "toolName": "search_repositories",
                    "params": {
                        "query": query,
                        "sort": sort,
                        "order": order,
                        "limit": limit
                    }
                }
            )
            
            if response.status_code != 200:
                return f"Error: Unable to search repositories (Status code: {response.status_code})"
                
            result = response.json()
            
            if not result.get("success", False):
                return f"Error: {result.get('error', 'Unknown error')}"
                
            # Format the results as a readable string
            repositories = result.get("repositories", [])
            total_count = result.get("total_count", 0)
            
            output = f"Search query: \"{result.get('query')}\"\n"
            output += f"Found {total_count} repositories (showing top {len(repositories)})\n\n"
            
            for i, repo in enumerate(repositories, 1):
                output += f"{i}. {repo.get('full_name')}\n"
                output += f"   URL: {repo.get('url')}\n"
                output += f"   Description: {repo.get('description') or 'No description'}\n"
                output += f"   Stars: {repo.get('stars', 0)}  |  Forks: {repo.get('forks', 0)}  |  Language: {repo.get('language') or 'Unknown'}\n"
                output += f"   Updated: {repo.get('updated_at')}\n\n"
                
            return output
            
        except Exception as e:
            return f"Error: {str(e)}"

class GetRepositoryDetailsTool(BaseTool):
    """Tool for getting detailed GitHub repository information using the GitHub MCP server."""
    
    name = "get_github_repository_details"
    description = """Get detailed information about a specific GitHub repository.
    Useful for learning about a project's features, popularity, and structure.
    Input should be a string representing the repository name in the format 'owner/repo'."""
    
    def _run(self, repo_name: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run the tool to get repository details."""
        try:
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers={"Content-Type": "application/json"},
                json={
                    "serverName": "github",
                    "toolName": "get_repository_details",
                    "params": {"repo_name": repo_name}
                }
            )
            
            if response.status_code != 200:
                return f"Error: Unable to get repository details (Status code: {response.status_code})"
                
            result = response.json()
            
            if not result.get("success", False):
                return f"Error: {result.get('error', 'Unknown error')}"
                
            # Format the results as a readable string
            repo = result.get("repository", {})
            
            output = f"Repository: {repo.get('full_name')}\n"
            output += f"URL: {repo.get('url')}\n\n"
            
            output += f"Description: {repo.get('description') or 'No description'}\n\n"
            
            output += f"Owner: {repo.get('owner', {}).get('login')}\n"
            output += f"Default branch: {repo.get('default_branch')}\n"
            output += f"License: {repo.get('license') or 'None specified'}\n"
            
            # Stats
            output += "\nStatistics:\n"
            output += f"- Stars: {repo.get('stars', 0)}\n"
            output += f"- Forks: {repo.get('forks', 0)}\n"
            output += f"- Watchers: {repo.get('watchers', 0)}\n"
            output += f"- Open Issues: {repo.get('open_issues', 0)}\n"
            
            # Languages
            languages = repo.get('languages', {})
            if languages:
                output += "\nLanguages:\n"
                for lang, bytes_count in languages.items():
                    output += f"- {lang}: {bytes_count} bytes\n"
                    
            # Topics
            topics = repo.get('topics', [])
            if topics:
                output += "\nTopics: " + ", ".join(topics) + "\n"
                
            # Dates
            output += f"\nCreated: {repo.get('created_at')}\n"
            output += f"Last updated: {repo.get('updated_at')}\n"
            output += f"Last pushed: {repo.get('pushed_at')}\n"
                
            return output
            
        except Exception as e:
            return f"Error: {str(e)}"

class ListRepositoryContentsTool(BaseTool):
    """Tool for listing GitHub repository contents using the GitHub MCP server."""
    
    name = "list_github_repository_contents"
    description = """List the contents of a GitHub repository directory.
    Useful for exploring project structure, finding specific files, or understanding code organization.
    Input should be a JSON string with 'repo_name', optional 'path' (defaults to root), and optional 'ref' (branch, tag, commit) fields."""
    
    def _run(self, tool_input: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run the tool to list repository contents."""
        try:
            # Parse input
            if tool_input.startswith("{") and tool_input.endswith("}"):
                input_data = json.loads(tool_input)
                repo_name = input_data.get("repo_name")
                path = input_data.get("path", "")
                ref = input_data.get("ref")
            else:
                repo_name = tool_input
                path = ""
                ref = None
                
            if not repo_name:
                return "Error: Repository name must be provided in format 'owner/repo'"
                
            params = {"repo_name": repo_name, "path": path}
            if ref:
                params["ref"] = ref
                
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers={"Content-Type": "application/json"},
                json={
                    "serverName": "github",
                    "toolName": "list_repository_contents",
                    "params": params
                }
            )
            
            if response.status_code != 200:
                return f"Error: Unable to list repository contents (Status code: {response.status_code})"
                
            result = response.json()
            
            if not result.get("success", False):
                return f"Error: {result.get('error', 'Unknown error')}"
                
            # Format the results as a readable string
            contents = result.get("contents", [])
            
            path_display = f"/{path}" if path else "/"
            branch_display = f" ({ref})" if ref else ""
            
            output = f"Contents of {result.get('repository')}{path_display}{branch_display}:\n\n"
            
            # Separate directories and files
            directories = [item for item in contents if item.get("type") == "dir"]
            files = [item for item in contents if item.get("type") == "file"]
            
            # List directories first
            if directories:
                output += "Directories:\n"
                for dir_item in sorted(directories, key=lambda x: x.get("name", "")):
                    output += f"- {dir_item.get('name')}/\n"
                output += "\n"
                
            # Then list files
            if files:
                output += "Files:\n"
                for file_item in sorted(files, key=lambda x: x.get("name", "")):
                    size = file_item.get("size", 0)
                    size_str = f"{size} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                    output += f"- {file_item.get('name')} ({size_str})\n"
                    
            if not contents:
                output += "No files or directories found.\n"
                
            return output
            
        except json.JSONDecodeError:
            # Try treating the input as just a repo name
            return self._run(json.dumps({"repo_name": tool_input}))
        except Exception as e:
            return f"Error: {str(e)}"

class GetFileContentTool(BaseTool):
    """Tool for fetching file content from GitHub repositories using the GitHub MCP server."""
    
    name = "get_github_file_content"
    description = """Get the content of a specific file from a GitHub repository.
    Useful for examining code, documentation, or data files.
    Input should be a JSON string with 'repo_name', 'file_path', and optional 'ref' (branch, tag, commit) fields."""
    
    def _run(self, tool_input: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run the tool to get file content."""
        try:
            # Parse input
            input_data = json.loads(tool_input)
            repo_name = input_data.get("repo_name")
            file_path = input_data.get("file_path")
            ref = input_data.get("ref")
            
            if not repo_name or not file_path:
                return "Error: Required parameters 'repo_name' and 'file_path' must be provided"
                
            params = {"repo_name": repo_name, "file_path": file_path}
            if ref:
                params["ref"] = ref
                
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers={"Content-Type": "application/json"},
                json={
                    "serverName": "github",
                    "toolName": "get_file_content",
                    "params": params
                }
            )
            
            if response.status_code != 200:
                return f"Error: Unable to get file content (Status code: {response.status_code})"
                
            result = response.json()
            
            if not result.get("success", False):
                return f"Error: {result.get('error', 'Unknown error')}"
                
            # Return the file content with some basic metadata
            file_size = result.get("size", 0)
            size_str = f"{file_size} bytes" if file_size < 1024 else f"{file_size/1024:.1f} KB"
            
            output = f"File: {result.get('path')} ({size_str})\n"
            output += f"Repository: {result.get('repository')}\n"
            output += f"URL: {result.get('url')}\n\n"
            output += "Content:\n\n"
            output += result.get("content", "")
                
            return output
            
        except json.JSONDecodeError:
            return "Error: Input must be a valid JSON string with 'repo_name' and 'file_path'"
        except Exception as e:
            return f"Error: {str(e)}"

class SearchCodeTool(BaseTool):
    """Tool for searching code in GitHub repositories using the GitHub MCP server."""
    
    name = "search_github_code"
    description = """Search for code in GitHub repositories.
    Useful for finding implementations, examples, or specific code patterns.
    Input should be a JSON string with 'query', and optional 'language', 'repo', and 'limit' fields."""
    
    def _run(self, tool_input: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run the tool to search code."""
        try:
            # Parse input - handle both string and JSON
            language = None
            repo = None
            limit = 10
            
            if tool_input.startswith("{") and tool_input.endswith("}"):
                try:
                    input_data = json.loads(tool_input)
                    query = input_data.get("query")
                    language = input_data.get("language")
                    repo = input_data.get("repo")
                    limit = input_data.get("limit", limit)
                except:
                    query = tool_input
            else:
                query = tool_input
                
            if not query:
                return "Error: Search query must be provided"
                
            params = {"query": query, "limit": limit}
            if language:
                params["language"] = language
            if repo:
                params["repo"] = repo
                
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers={"Content-Type": "application/json"},
                json={
                    "serverName": "github",
                    "toolName": "search_code",
                    "params": params
                }
            )
            
            if response.status_code != 200:
                return f"Error: Unable to search code (Status code: {response.status_code})"
                
            result = response.json()
            
            if not result.get("success", False):
                return f"Error: {result.get('error', 'Unknown error')}"
                
            # Format the results as a readable string
            results = result.get("results", [])
            total_count = result.get("total_count", 0)
            
            output = f"Code search query: \"{result.get('query')}\"\n"
            if result.get("language"):
                output += f"Language filter: {result.get('language')}\n"
            if result.get("repository"):
                output += f"Repository filter: {result.get('repository')}\n"
                
            output += f"Found {total_count} results (showing top {len(results)})\n\n"
            
            for i, item in enumerate(results, 1):
                repo_info = item.get("repository", {})
                output += f"{i}. {repo_info.get('full_name')}: {item.get('path')}\n"
                output += f"   URL: {item.get('url')}\n\n"
                
            return output
            
        except Exception as e:
            return f"Error: {str(e)}"

class ListIssuestTool(BaseTool):
    """Tool for listing GitHub issues using the GitHub MCP server."""
    
    name = "list_github_issues"
    description = """List issues in a GitHub repository.
    Useful for project management, bug tracking, or contributing to open source.
    Input should be a string with the repository name or a JSON string with 'repo_name', and optional 'state', 'sort', 'direction', and 'limit' fields."""
    
    def _run(self, tool_input: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run the tool to list issues."""
        try:
            # Parse input - handle both string and JSON
            state = "open"
            sort = "created"
            direction = "desc"
            limit = 10
            
            if tool_input.startswith("{") and tool_input.endswith("}"):
                try:
                    input_data = json.loads(tool_input)
                    repo_name = input_data.get("repo_name")
                    state = input_data.get("state", state)
                    sort = input_data.get("sort", sort)
                    direction = input_data.get("direction", direction)
                    limit = input_data.get("limit", limit)
                except:
                    repo_name = tool_input
            else:
                repo_name = tool_input
                
            if not repo_name:
                return "Error: Repository name must be provided in format 'owner/repo'"
                
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers={"Content-Type": "application/json"},
                json={
                    "serverName": "github",
                    "toolName": "list_issues",
                    "params": {
                        "repo_name": repo_name,
                        "state": state,
                        "sort": sort,
                        "direction": direction,
                        "limit": limit
                    }
                }
            )
            
            if response.status_code != 200:
                return f"Error: Unable to list issues (Status code: {response.status_code})"
                
            result = response.json()
            
            if not result.get("success", False):
                return f"Error: {result.get('error', 'Unknown error')}"
                
            # Format the results as a readable string
            issues = result.get("issues", [])
            
            output = f"Issues for {result.get('repository')} (state: {result.get('state_filter')}):\n"
            output += f"Found {len(issues)} issues\n\n"
            
            for i, issue in enumerate(issues, 1):
                output += f"{i}. #{issue.get('number')} - {issue.get('title')}\n"
                output += f"   State: {issue.get('state')}  |  Comments: {issue.get('comments')}\n"
                output += f"   Created: {issue.get('created_at')}  |  Updated: {issue.get('updated_at')}\n"
                
                # Labels
                labels = issue.get("labels", [])
                if labels:
                    output += f"   Labels: {', '.join(labels)}\n"
                    
                output += f"   URL: {issue.get('url')}\n\n"
                
            if not issues:
                output += "No issues found.\n"
                
            return output
            
        except Exception as e:
            return f"Error: {str(e)}"

class GetIssueDetailsTool(BaseTool):
    """Tool for getting detailed information about GitHub issues using the GitHub MCP server."""
    
    name = "get_github_issue_details"
    description = """Get detailed information about a specific GitHub issue.
    Useful for understanding bug reports, feature requests, or discussions.
    Input should be a JSON string with 'repo_name' and 'issue_number' fields."""
    
    def _run(self, tool_input: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run the tool to get issue details."""
        try:
            # Parse input
            if tool_input.startswith("{") and tool_input.endswith("}"):
                input_data = json.loads(tool_input)
                repo_name = input_data.get("repo_name")
                issue_number = input_data.get("issue_number")
            else:
                # Try to parse repo_name#issue_number format
                parts = tool_input.split('#')
                if len(parts) == 2:
                    repo_name = parts[0].strip()
                    try:
                        issue_number = int(parts[1].strip())
                    except:
                        return "Error: Issue number must be an integer"
                else:
                    return "Error: Input must be a JSON string with 'repo_name' and 'issue_number' or format 'owner/repo#123'"
                
            if not repo_name or not issue_number:
                return "Error: Required parameters 'repo_name' and 'issue_number' must be provided"
                
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers={"Content-Type": "application/json"},
                json={
                    "serverName": "github",
                    "toolName": "get_issue_details",
                    "params": {
                        "repo_name": repo_name,
                        "issue_number": issue_number
                    }
                }
            )
            
            if response.status_code != 200:
                return f"Error: Unable to get issue details (Status code: {response.status_code})"
                
            result = response.json()
            
            if not result.get("success", False):
                return f"Error: {result.get('error', 'Unknown error')}"
                
            # Format the results as a readable string
            issue = result.get("issue", {})
            
            output = f"Issue #{issue.get('number')}: {issue.get('title')}\n"
            output += f"Repository: {result.get('repository')}\n"
            output += f"State: {issue.get('state')}\n"
            output += f"URL: {issue.get('url')}\n\n"
            
            # Author info
            output += f"Author: {issue.get('author')}\n"
            output += f"Created: {issue.get('created_at')}\n"
            output += f"Updated: {issue.get('updated_at')}\n"
            
            if issue.get('closed_at'):
                output += f"Closed: {issue.get('closed_at')}\n"
                
            # Labels and assignees
            labels = issue.get("labels", [])
            if labels:
                output += f"Labels: {', '.join(labels)}\n"
                
            assignees = issue.get("assignees", [])
            if assignees:
                output += f"Assignees: {', '.join(assignees)}\n"
                
            # Body content
            output += "\nDescription:\n"
            output += f"{issue.get('body') or 'No description provided.'}\n\n"
            
            # Comments
            comments = issue.get("comments", [])
            if comments:
                output += f"Comments ({issue.get('comments_count', len(comments))}):\n\n"
                
                for i, comment in enumerate(comments, 1):
                    output += f"Comment {i} by {comment.get('author')} on {comment.get('created_at')}:\n"
                    output += f"{comment.get('body')}\n\n"
            else:
                output += "No comments.\n"
                
            return output
            
        except json.JSONDecodeError:
            # Already handled above for the repo_name#issue_number format
            return "Error: Input must be a valid JSON string with 'repo_name' and 'issue_number' or format 'owner/repo#123'"
        except Exception as e:
            return f"Error: {str(e)}"
