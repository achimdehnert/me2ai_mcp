"""
Tests for the ME2AI MCP authentication system.
"""
import os
import pytest
import time
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from me2ai_mcp.auth import AuthManager, AuthResult, TokenAuth, APIKeyAuth


class TestAuthManager:
    """Tests for the AuthManager class."""

    def test_should_initialize_with_empty_providers_when_default_constructor_used(self):
        """Test that AuthManager initializes with empty providers list."""
        auth_manager = AuthManager()
        assert auth_manager.providers == []
        assert auth_manager.has_token() is False

    def test_should_add_provider_successfully_when_valid_provider_given(self):
        """Test adding a provider to AuthManager."""
        auth_manager = AuthManager()
        provider = APIKeyAuth("test-key")
        auth_manager.add_provider(provider)
        assert len(auth_manager.providers) == 1
        assert auth_manager.providers[0] == provider

    def test_should_authenticate_successfully_when_valid_credentials_provided(self):
        """Test successful authentication."""
        auth_manager = AuthManager()
        auth_manager.add_provider(APIKeyAuth("test-key"))
        
        # Mock request with valid API key
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer test-key"}
        
        result = auth_manager.authenticate(mock_request)
        assert result.authenticated is True
        assert result.provider_name == "api_key"

    def test_should_fail_authentication_when_invalid_credentials_provided(self):
        """Test failed authentication with invalid credentials."""
        auth_manager = AuthManager()
        auth_manager.add_provider(APIKeyAuth("test-key"))
        
        # Mock request with invalid API key
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer wrong-key"}
        
        result = auth_manager.authenticate(mock_request)
        assert result.authenticated is False

    def test_should_try_all_providers_when_multiple_providers_configured(self):
        """Test that all providers are tried during authentication."""
        auth_manager = AuthManager()
        auth_manager.add_provider(APIKeyAuth("api-key"))
        auth_manager.add_provider(TokenAuth("token-value"))
        
        # Mock request that should match the second provider
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Token token-value"}
        
        result = auth_manager.authenticate(mock_request)
        assert result.authenticated is True
        assert result.provider_name == "token"

    def test_should_return_token_when_has_token_and_token_provider_exists(self):
        """Test has_token and get_token methods."""
        auth_manager = AuthManager()
        token_auth = TokenAuth("test-token")
        auth_manager.add_provider(token_auth)
        
        assert auth_manager.has_token() is True
        assert auth_manager.get_token() == token_auth

    @patch.dict(os.environ, {"GITHUB_TOKEN": "github-test-token"})
    def test_should_create_from_github_token_when_environment_variable_exists(self):
        """Test from_github_token factory method with environment variable."""
        auth_manager = AuthManager.from_github_token()
        
        assert auth_manager.has_token() is True
        token_auth = auth_manager.get_token()
        assert token_auth.token == "github-test-token"
        
        # Test with mock request
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Token github-test-token"}
        result = auth_manager.authenticate(mock_request)
        assert result.authenticated is True

    @patch.dict(os.environ, {}, clear=True)
    def test_should_create_empty_manager_when_no_github_token_in_environment(self):
        """Test from_github_token factory method without environment variable."""
        auth_manager = AuthManager.from_github_token()
        assert auth_manager.has_token() is False
        assert len(auth_manager.providers) == 0


class TestTokenAuth:
    """Tests for the TokenAuth class."""

    def test_should_initialize_with_token_when_provided(self):
        """Test TokenAuth initialization."""
        token_auth = TokenAuth("test-token")
        assert token_auth.token == "test-token"
        assert token_auth.expiration is None
        assert token_auth.header_name == "Authorization"
        assert token_auth.token_type == "Token"

    def test_should_authenticate_when_valid_token_in_header(self):
        """Test successful token authentication."""
        token_auth = TokenAuth("test-token")
        
        # Valid token in header
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Token test-token"}
        
        result = token_auth.authenticate(mock_request)
        assert result.authenticated is True
        assert result.provider_name == "token"

    def test_should_authenticate_when_valid_token_in_custom_header(self):
        """Test token authentication with custom header."""
        token_auth = TokenAuth(
            token="test-token",
            header_name="X-API-Token",
            token_type=""
        )
        
        # Valid token in custom header
        mock_request = MagicMock()
        mock_request.headers = {"X-API-Token": "test-token"}
        
        result = token_auth.authenticate(mock_request)
        assert result.authenticated is True

    def test_should_fail_when_token_expired(self):
        """Test authentication with expired token."""
        # Create token that expired 1 hour ago
        expiration = datetime.now() - timedelta(hours=1)
        token_auth = TokenAuth("test-token", expiration=expiration)
        
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Token test-token"}
        
        result = token_auth.authenticate(mock_request)
        assert result.authenticated is False

    def test_should_succeed_when_token_not_expired(self):
        """Test authentication with valid non-expired token."""
        # Create token that expires 1 hour from now
        expiration = datetime.now() + timedelta(hours=1)
        token_auth = TokenAuth("test-token", expiration=expiration)
        
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Token test-token"}
        
        result = token_auth.authenticate(mock_request)
        assert result.authenticated is True

    def test_should_fail_when_wrong_token_format(self):
        """Test authentication with wrong token format."""
        token_auth = TokenAuth("test-token")
        
        # Wrong format (missing token type)
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "test-token"}
        
        result = token_auth.authenticate(mock_request)
        assert result.authenticated is False


class TestAPIKeyAuth:
    """Tests for the APIKeyAuth class."""

    def test_should_initialize_with_api_key_when_provided(self):
        """Test APIKeyAuth initialization."""
        api_key_auth = APIKeyAuth("test-key")
        assert api_key_auth.api_key == "test-key"
        assert api_key_auth.header_name == "Authorization"
        assert api_key_auth.key_type == "Bearer"

    def test_should_authenticate_when_valid_api_key_in_header(self):
        """Test successful API key authentication."""
        api_key_auth = APIKeyAuth("test-key")
        
        # Valid API key in header
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer test-key"}
        
        result = api_key_auth.authenticate(mock_request)
        assert result.authenticated is True
        assert result.provider_name == "api_key"

    def test_should_authenticate_when_valid_api_key_in_custom_header(self):
        """Test API key authentication with custom header."""
        api_key_auth = APIKeyAuth(
            api_key="test-key",
            header_name="X-API-Key",
            key_type=""
        )
        
        # Valid API key in custom header
        mock_request = MagicMock()
        mock_request.headers = {"X-API-Key": "test-key"}
        
        result = api_key_auth.authenticate(mock_request)
        assert result.authenticated is True

    def test_should_fail_when_wrong_api_key(self):
        """Test authentication with wrong API key."""
        api_key_auth = APIKeyAuth("test-key")
        
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer wrong-key"}
        
        result = api_key_auth.authenticate(mock_request)
        assert result.authenticated is False

    def test_should_fail_when_header_missing(self):
        """Test authentication with missing header."""
        api_key_auth = APIKeyAuth("test-key")
        
        # No Authorization header
        mock_request = MagicMock()
        mock_request.headers = {}
        
        result = api_key_auth.authenticate(mock_request)
        assert result.authenticated is False
