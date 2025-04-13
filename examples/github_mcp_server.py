"""
GitHub Integration ME2AI MCP Server Example

This example demonstrates how to create an MCP server
that integrates with GitHub using the ME2AI MCP framework.
"""

import os
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from me2ai_mcp.base import ME2AIMCPServer
from me2ai_mcp.auth import AuthManager


def main():
    """Run a GitHub-integrated ME2AI MCP server demonstration."""
    # Load environment variables for tokens
    load_dotenv()
    
    # Create a new MCP server instance
    server = ME2AIMCPServer(
        server_name="github_mcp_server",
        description="GitHub Integration ME2AI MCP Server",
        version="0.0.5"
    )
    
    print(f"Starting {server.description} (v{server.version})")
    
    # Get authentication token
    auth_manager = AuthManager()
    github_token = auth_manager.get_token(
        token_var_names=["GITHUB_API_KEY", "GITHUB_TOKEN", "GITHUB_ACCESS_TOKEN"]
    )
    
    if not github_token:
        print("WARNING: No GitHub token found. Some functionality will be limited.")
        print("Set GITHUB_API_KEY, GITHUB_TOKEN, or GITHUB_ACCESS_TOKEN environment variable.")
    
    # Register GitHub repository search tool
    @server.register_tool
    def search_github_repos(
        query: str,
        token: str = github_token,
        language: Optional[str] = None,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search GitHub repositories based on a query.
        
        Args:
            query: Search query string
            token: GitHub API token (optional if set in environment)
            language: Filter by programming language
            max_results: Maximum number of results to return
            
        Returns:
            Dict containing search results
        """
        import requests
        
        # Build query string
        search_query = query
        if language:
            search_query += f" language:{language}"
        
        # Set up headers with token if available
        headers = {}
        if token:
            headers["Authorization"] = f"token {token}"
        
        # Make GitHub API request
        response = None
        try:
            api_url = "https://api.github.com/search/repositories"
            params = {
                "q": search_query,
                "sort": "stars",
                "order": "desc",
                "per_page": max_results
            }
            
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant information
            results = []
            for repo in data.get("items", [])[:max_results]:
                results.append({
                    "name": repo["name"],
                    "full_name": repo["full_name"],
                    "description": repo["description"],
                    "url": repo["html_url"],
                    "stars": repo["stargazers_count"],
                    "forks": repo["forks_count"],
                    "language": repo["language"]
                })
                
            return {
                "query": query,
                "language_filter": language,
                "result_count": len(results),
                "total_count": data.get("total_count", 0),
                "results": results
            }
            
        except Exception as e:
            # Error handling with detailed information
            error_details = {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            }
            
            # Add response details if available
            if response:
                error_details["status_code"] = response.status_code
                try:
                    error_details["response_body"] = response.json()
                except:
                    error_details["response_body"] = response.text
            
            return {"error": "Failed to search GitHub repositories", "details": error_details}
    
    # Register GitHub repository contents tool
    @server.register_tool
    def list_github_repo_contents(
        repo_full_name: str,
        path: str = "",
        token: str = github_token,
        ref: str = "main"
    ) -> Dict[str, Any]:
        """
        List contents of a GitHub repository directory.
        
        Args:
            repo_full_name: Full repository name (e.g., "username/repo")
            path: Directory path within repository (empty for root)
            token: GitHub API token (optional if set in environment)
            ref: Branch, tag, or commit SHA
            
        Returns:
            Dict containing directory contents
        """
        import requests
        
        # Set up headers with token if available
        headers = {}
        if token:
            headers["Authorization"] = f"token {token}"
        
        # Make GitHub API request
        response = None
        try:
            api_url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
            params = {"ref": ref}
            
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Process results
            contents = []
            
            # Handle array response (directory listing)
            if isinstance(data, list):
                for item in data:
                    contents.append({
                        "name": item["name"],
                        "path": item["path"],
                        "type": item["type"],
                        "size": item.get("size", 0),
                        "url": item["html_url"]
                    })
            # Handle object response (single file)
            else:
                contents.append({
                    "name": data["name"],
                    "path": data["path"],
                    "type": data["type"],
                    "size": data.get("size", 0),
                    "url": data["html_url"]
                })
                
            return {
                "repo": repo_full_name,
                "path": path,
                "ref": ref,
                "contents": contents
            }
            
        except Exception as e:
            # Error handling with detailed information
            error_details = {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            }
            
            # Add response details if available
            if response:
                error_details["status_code"] = response.status_code
                try:
                    error_details["response_body"] = response.json()
                except:
                    error_details["response_body"] = response.text
            
            return {"error": "Failed to list repository contents", "details": error_details}
    
    # Demo execution
    print("\n--- GitHub API Demo ---")
    print("(Note: Some features require GitHub API token)")
    
    if github_token:
        print("\nExecuting 'search_github_repos' tool:")
        search_result = server.execute_tool("search_github_repos", {
            "query": "mcp protocol",
            "language": "python",
            "max_results": 3
        })
        print(f"Result: {json.dumps(search_result, indent=2)}")
        
        # If we found results, explore one repository
        if "results" in search_result and search_result["results"]:
            first_repo = search_result["results"][0]["full_name"]
            print(f"\nExecuting 'list_github_repo_contents' tool for {first_repo}:")
            contents_result = server.execute_tool("list_github_repo_contents", {
                "repo_full_name": first_repo
            })
            print(f"Result: {json.dumps(contents_result, indent=2)}")
    else:
        print("\nSkipping GitHub API calls - no token available")
        print("To run this demo, set GITHUB_API_KEY in your environment or .env file")
    
    # List available tools
    print("\nAvailable GitHub integration tools:")
    tools = server.get_tools()
    for tool in tools:
        print(f"- {tool['name']}: {tool['description']}")


if __name__ == "__main__":
    main()
