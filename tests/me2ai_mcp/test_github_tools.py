"""
Tests for the ME2AI MCP GitHub tools including authentication edge cases.
"""
import pytest
import os
import json
from unittest.mock import patch, MagicMock, Mock
import asyncio
import requests

from me2ai_mcp.tools.github import (
    GitHubRepositoryTool,
    GitHubCodeTool,
    GitHubIssuesTool
)
from me2ai_mcp.auth import AuthManager


class TestGitHubTokenHandling:
    """Tests for GitHub API token handling."""

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    def test_should_use_token_when_available_in_environment(self):
        """Test token detection from environment variables."""
        auth_manager = AuthManager.from_github_token()
        
        assert auth_manager.has_token() is True
        token = auth_manager.get_token()
        assert token.token == "test-token"

    @patch.dict(os.environ, {
        "GITHUB_API_KEY": "test-api-key",
        "GITHUB_TOKEN": "test-token"
    })
    def test_should_prioritize_github_api_key_when_multiple_variables_available(self):
        """Test token priority when multiple environment variables are set."""
        auth_manager = AuthManager.from_github_token()
        
        assert auth_manager.has_token() is True
        token = auth_manager.get_token()
        assert token.token == "test-api-key"  # Should use GITHUB_API_KEY first

    @patch.dict(os.environ, {}, clear=True)
    def test_should_create_auth_manager_without_token_when_no_variables_set(self):
        """Test behavior when no GitHub token is available."""
        auth_manager = AuthManager.from_github_token()
        
        assert auth_manager.has_token() is False
        assert len(auth_manager.providers) == 0


class TestGitHubRepositoryTool:
    """Tests for the GitHubRepositoryTool class."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_github_repo_search_with_token(self):
        """Test GitHub repository search with a valid token."""
        tool = GitHubRepositoryTool()
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "full_name": "user/repo1",
                    "html_url": "https://github.com/user/repo1",
                    "description": "Test repo 1",
                    "stargazers_count": 100
                },
                {
                    "full_name": "user/repo2",
                    "html_url": "https://github.com/user/repo2",
                    "description": "Test repo 2",
                    "stargazers_count": 50
                }
            ],
            "total_count": 2
        }
        
        with patch('requests.get', return_value=mock_response):
            result = await tool.execute({
                "operation": "search",
                "query": "test repo"
            })
            
            # Should succeed with repositories
            assert result["success"] is True
            assert len(result["repositories"]) == 2
            assert result["repositories"][0]["name"] == "user/repo1"
            assert result["repositories"][1]["name"] == "user/repo2"
            
            # Headers should include authorization
            args, kwargs = requests.get.call_args
            headers = kwargs["headers"]
            assert "Authorization" in headers
            assert headers["Authorization"] == "token test-token"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_github_no_token_fallback(self):
        """Test GitHub tools fallback behavior without token."""
        tool = GitHubRepositoryTool()
        
        # Mock successful response but with limited results (as would happen without auth)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "full_name": "user/repo1",
                    "html_url": "https://github.com/user/repo1",
                    "description": "Test repo 1",
                    "stargazers_count": 100
                }
            ],
            "total_count": 1
        }
        
        with patch('requests.get', return_value=mock_response):
            result = await tool.execute({
                "operation": "search",
                "query": "test repo"
            })
            
            # Should succeed but with limited results and a warning
            assert result["success"] is True
            assert len(result["repositories"]) == 1
            assert "warning" in result
            assert "rate limit" in result["warning"].lower()
            
            # Headers should NOT include authorization
            args, kwargs = requests.get.call_args
            headers = kwargs["headers"]
            assert "Authorization" not in headers

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_github_rate_limit_handling(self):
        """Test handling of GitHub API rate limit errors."""
        tool = GitHubRepositoryTool()
        
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.json.return_value = {
            "message": "API rate limit exceeded",
            "documentation_url": "https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting"
        }
        
        with patch('requests.get', return_value=mock_response):
            with patch('requests.get', side_effect=requests.HTTPError("API rate limit exceeded")):
                result = await tool.execute({
                    "operation": "search",
                    "query": "test repo"
                })
                
                # Should fail with specific rate limit error
                assert result["success"] is False
                assert "rate limit" in result["error"].lower()
                assert result["exception_type"] == "HTTPError"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_github_repository_info(self):
        """Test getting detailed repository information."""
        tool = GitHubRepositoryTool()
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "full_name": "user/repo",
            "html_url": "https://github.com/user/repo",
            "description": "Test repository",
            "stargazers_count": 100,
            "forks_count": 20,
            "open_issues_count": 5,
            "language": "Python",
            "created_at": "2022-01-01T00:00:00Z",
            "updated_at": "2022-02-01T00:00:00Z",
            "topics": ["python", "testing", "github"]
        }
        
        with patch('requests.get', return_value=mock_response):
            result = await tool.execute({
                "operation": "info",
                "repository": "user/repo"
            })
            
            # Should succeed with repository details
            assert result["success"] is True
            assert result["repository"]["name"] == "user/repo"
            assert result["repository"]["stars"] == 100
            assert result["repository"]["forks"] == 20
            assert result["repository"]["language"] == "Python"
            assert len(result["repository"]["topics"]) == 3
            assert "python" in result["repository"]["topics"]


class TestGitHubCodeTool:
    """Tests for the GitHubCodeTool class."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_github_search_code(self):
        """Test GitHub code search."""
        tool = GitHubCodeTool()
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "name": "file1.py",
                    "path": "src/file1.py",
                    "repository": {
                        "full_name": "user/repo"
                    },
                    "html_url": "https://github.com/user/repo/blob/main/src/file1.py"
                },
                {
                    "name": "file2.py",
                    "path": "src/file2.py",
                    "repository": {
                        "full_name": "user/repo"
                    },
                    "html_url": "https://github.com/user/repo/blob/main/src/file2.py"
                }
            ],
            "total_count": 2
        }
        
        with patch('requests.get', return_value=mock_response):
            result = await tool.execute({
                "operation": "search",
                "query": "test function",
                "language": "python"
            })
            
            # Should succeed with code files
            assert result["success"] is True
            assert len(result["files"]) == 2
            assert result["files"][0]["name"] == "file1.py"
            assert result["files"][1]["name"] == "file2.py"
            assert result["files"][0]["repository"] == "user/repo"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_github_get_file_content(self):
        """Test getting file content from GitHub."""
        tool = GitHubCodeTool()
        
        # Mock successful response for content
        mock_content_response = Mock()
        mock_content_response.status_code = 200
        mock_content_response.json.return_value = {
            "content": "ZGVmIHRlc3RfZnVuY3Rpb24oKToKICAgIHJldHVybiAiSGVsbG8gV29ybGQiCg==",  # Base64 encoded
            "encoding": "base64"
        }
        
        with patch('requests.get', return_value=mock_content_response):
            result = await tool.execute({
                "operation": "content",
                "repository": "user/repo",
                "path": "src/file.py"
            })
            
            # Should succeed with decoded content
            assert result["success"] is True
            assert "def test_function():" in result["content"]
            assert "return \"Hello World\"" in result["content"]

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_github_get_large_file_content(self):
        """Test handling of large file content from GitHub."""
        tool = GitHubCodeTool(max_file_size=100)  # Small max size for testing
        
        # Mock response with large content
        mock_content_response = Mock()
        mock_content_response.status_code = 200
        # Generate a base64 string that would decode to more than max_file_size
        large_content = "A" * 200
        import base64
        encoded_content = base64.b64encode(large_content.encode()).decode()
        
        mock_content_response.json.return_value = {
            "content": encoded_content,
            "encoding": "base64",
            "size": 200  # Actual size before encoding
        }
        
        with patch('requests.get', return_value=mock_content_response):
            result = await tool.execute({
                "operation": "content",
                "repository": "user/repo",
                "path": "src/large_file.py"
            })
            
            # Should fail due to file size
            assert result["success"] is False
            assert "File too large" in result["error"]
            assert "200 bytes" in result["error"]

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "invalid-token"})
    async def test_github_authentication_failure(self):
        """Test handling of GitHub authentication failure."""
        tool = GitHubCodeTool()
        
        # Mock authentication failure response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "message": "Bad credentials",
            "documentation_url": "https://docs.github.com/rest"
        }
        
        with patch('requests.get', return_value=mock_response):
            with patch('requests.get', side_effect=requests.HTTPError("Bad credentials")):
                result = await tool.execute({
                    "operation": "search",
                    "query": "test function"
                })
                
                # Should fail with authentication error
                assert result["success"] is False
                assert "authentication" in result["error"].lower() or "credentials" in result["error"].lower()
                assert result["exception_type"] == "HTTPError"


class TestGitHubIssuesTool:
    """Tests for the GitHubIssuesTool class."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_github_list_issues(self):
        """Test listing GitHub issues."""
        tool = GitHubIssuesTool()
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "number": 1,
                "title": "Test Issue 1",
                "state": "open",
                "html_url": "https://github.com/user/repo/issues/1",
                "body": "Issue description 1",
                "created_at": "2022-01-01T00:00:00Z",
                "user": {
                    "login": "user1"
                },
                "labels": [
                    {"name": "bug"},
                    {"name": "high-priority"}
                ]
            },
            {
                "number": 2,
                "title": "Test Issue 2",
                "state": "closed",
                "html_url": "https://github.com/user/repo/issues/2",
                "body": "Issue description 2",
                "created_at": "2022-01-02T00:00:00Z",
                "user": {
                    "login": "user2"
                },
                "labels": [
                    {"name": "enhancement"}
                ]
            }
        ]
        
        with patch('requests.get', return_value=mock_response):
            result = await tool.execute({
                "operation": "list",
                "repository": "user/repo",
                "state": "all"
            })
            
            # Should succeed with issues
            assert result["success"] is True
            assert len(result["issues"]) == 2
            assert result["issues"][0]["number"] == 1
            assert result["issues"][0]["title"] == "Test Issue 1"
            assert result["issues"][0]["state"] == "open"
            assert len(result["issues"][0]["labels"]) == 2
            assert "bug" in result["issues"][0]["labels"]
            
            assert result["issues"][1]["number"] == 2
            assert result["issues"][1]["state"] == "closed"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_github_get_issue_details(self):
        """Test getting detailed GitHub issue information."""
        tool = GitHubIssuesTool()
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "number": 1,
            "title": "Test Issue",
            "state": "open",
            "html_url": "https://github.com/user/repo/issues/1",
            "body": "Issue description with details",
            "created_at": "2022-01-01T00:00:00Z",
            "updated_at": "2022-01-02T00:00:00Z",
            "user": {
                "login": "user1",
                "html_url": "https://github.com/user1"
            },
            "labels": [
                {"name": "bug"},
                {"name": "high-priority"}
            ],
            "comments": 5
        }
        
        # Mock comments response
        mock_comments_response = Mock()
        mock_comments_response.status_code = 200
        mock_comments_response.json.return_value = [
            {
                "body": "Comment 1",
                "created_at": "2022-01-01T01:00:00Z",
                "user": {
                    "login": "user2"
                }
            },
            {
                "body": "Comment 2",
                "created_at": "2022-01-01T02:00:00Z",
                "user": {
                    "login": "user1"
                }
            }
        ]
        
        with patch('requests.get', side_effect=[mock_response, mock_comments_response]):
            result = await tool.execute({
                "operation": "detail",
                "repository": "user/repo",
                "issue_number": 1,
                "include_comments": True
            })
            
            # Should succeed with issue details and comments
            assert result["success"] is True
            assert result["issue"]["number"] == 1
            assert result["issue"]["title"] == "Test Issue"
            assert result["issue"]["state"] == "open"
            assert "Issue description with details" in result["issue"]["body"]
            
            # Check comments
            assert "comments" in result
            assert len(result["comments"]) == 2
            assert result["comments"][0]["body"] == "Comment 1"
            assert result["comments"][1]["body"] == "Comment 2"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_github_create_issue(self):
        """Test creating a GitHub issue."""
        tool = GitHubIssuesTool()
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 201  # Created
        mock_response.json.return_value = {
            "number": 3,
            "title": "New Issue",
            "html_url": "https://github.com/user/repo/issues/3"
        }
        
        with patch('requests.post', return_value=mock_response):
            result = await tool.execute({
                "operation": "create",
                "repository": "user/repo",
                "title": "New Issue",
                "body": "Issue description",
                "labels": ["bug", "documentation"]
            })
            
            # Should succeed with new issue details
            assert result["success"] is True
            assert result["issue"]["number"] == 3
            assert result["issue"]["title"] == "New Issue"
            
            # Check request payload
            args, kwargs = requests.post.call_args
            data = json.loads(kwargs["data"])
            assert data["title"] == "New Issue"
            assert data["body"] == "Issue description"
            assert "bug" in data["labels"]
            assert "documentation" in data["labels"]

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_github_issues_no_token_error(self):
        """Test error handling when trying to create issues without a token."""
        tool = GitHubIssuesTool()
        
        result = await tool.execute({
            "operation": "create",
            "repository": "user/repo",
            "title": "New Issue",
            "body": "Issue description"
        })
        
        # Should fail with authentication error
        assert result["success"] is False
        assert "authentication" in result["error"].lower() or "token" in result["error"].lower()
        assert "GitHub token is required" in result["error"]
