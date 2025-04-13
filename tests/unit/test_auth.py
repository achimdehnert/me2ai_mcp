"""
Unit tests for the ME2AI MCP authentication module.

These tests verify the functionality of the authentication providers
and the authentication manager.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from me2ai_mcp.auth import AuthProvider, APIKeyAuth, TokenAuth, AuthManager


class TestAPIKeyAuth:
    """Tests for the APIKeyAuth class."""

    def test_should_authenticate_with_valid_api_key(self) -> None:
        """Test successful authentication with a valid API key."""
        # Arrange
        auth = APIKeyAuth(api_key="test-key")
        credentials = {"X-API-Key": "test-key"}
        
        # Act
        result = auth.authenticate(credentials)
        
        # Assert
        assert result is True

    def test_should_fail_authentication_with_invalid_api_key(self) -> None:
        """Test failed authentication with an invalid API key."""
        # Arrange
        auth = APIKeyAuth(api_key="test-key")
        credentials = {"X-API-Key": "wrong-key"}
        
        # Act
        result = auth.authenticate(credentials)
        
        # Assert
        assert result is False

    def test_should_fail_authentication_with_missing_api_key(self) -> None:
        """Test failed authentication with missing API key in credentials."""
        # Arrange
        auth = APIKeyAuth(api_key="test-key")
        credentials = {}
        
        # Act
        result = auth.authenticate(credentials)
        
        # Assert
        assert result is False

    def test_should_initialize_from_environment_variable(self) -> None:
        """Test initialization of API key from environment variable."""
        # Arrange
        with patch.dict(os.environ, {"TEST_API_KEY": "env-test-key"}):
            # Act
            auth = APIKeyAuth(env_var_name="TEST_API_KEY")
            
            # Assert
            assert auth.api_key == "env-test-key"
            assert auth.get_auth_headers() == {"X-API-Key": "env-test-key"}

    def test_should_handle_missing_environment_variable(self) -> None:
        """Test handling of missing environment variable."""
        # Arrange & Act
        with patch.dict(os.environ, {}, clear=True):
            auth = APIKeyAuth(env_var_name="NONEXISTENT_KEY")
            
            # Assert
            assert auth.api_key is None
            assert auth.get_auth_headers() == {}


class TestTokenAuth:
    """Tests for the TokenAuth class."""

    def test_should_authenticate_with_valid_token(self) -> None:
        """Test successful authentication with a valid token."""
        # Arrange
        auth = TokenAuth(token="test-token")
        credentials = {"Authorization": "Bearer test-token"}
        
        # Act
        result = auth.authenticate(credentials)
        
        # Assert
        assert result is True

    def test_should_fail_authentication_with_invalid_token(self) -> None:
        """Test failed authentication with an invalid token."""
        # Arrange
        auth = TokenAuth(token="test-token")
        credentials = {"Authorization": "Bearer wrong-token"}
        
        # Act
        result = auth.authenticate(credentials)
        
        # Assert
        assert result is False

    def test_should_fail_authentication_with_missing_token(self) -> None:
        """Test failed authentication with missing token in credentials."""
        # Arrange
        auth = TokenAuth(token="test-token")
        credentials = {}
        
        # Act
        result = auth.authenticate(credentials)
        
        # Assert
        assert result is False

    def test_should_fail_authentication_with_wrong_auth_scheme(self) -> None:
        """Test failed authentication with wrong authentication scheme."""
        # Arrange
        auth = TokenAuth(token="test-token", auth_scheme="Bearer")
        credentials = {"Authorization": "Basic test-token"}
        
        # Act
        result = auth.authenticate(credentials)
        
        # Assert
        assert result is False

    def test_should_initialize_from_environment_variable(self) -> None:
        """Test initialization of token from environment variable."""
        # Arrange
        with patch.dict(os.environ, {"TEST_TOKEN": "env-test-token"}):
            # Act
            auth = TokenAuth(env_var_name="TEST_TOKEN")
            
            # Assert
            assert auth.token == "env-test-token"
            assert auth.get_auth_headers() == {"Authorization": "Bearer env-test-token"}

    def test_should_use_custom_auth_scheme(self) -> None:
        """Test using a custom authentication scheme."""
        # Arrange
        auth = TokenAuth(token="test-token", auth_scheme="API-Key")
        
        # Act & Assert
        assert auth.get_auth_headers() == {"Authorization": "API-Key test-token"}


class TestAuthManager:
    """Tests for the AuthManager class."""

    def test_should_authenticate_when_any_provider_succeeds(self) -> None:
        """Test successful authentication when any provider succeeds."""
        # Arrange
        provider1 = MagicMock(spec=AuthProvider)
        provider1.authenticate.return_value = False
        
        provider2 = MagicMock(spec=AuthProvider)
        provider2.authenticate.return_value = True
        
        auth_manager = AuthManager([provider1, provider2])
        credentials = {"test": "credentials"}
        
        # Act
        result = auth_manager.authenticate(credentials)
        
        # Assert
        assert result is True
        provider1.authenticate.assert_called_once_with(credentials)
        provider2.authenticate.assert_called_once_with(credentials)

    def test_should_fail_authentication_when_all_providers_fail(self) -> None:
        """Test failed authentication when all providers fail."""
        # Arrange
        provider1 = MagicMock(spec=AuthProvider)
        provider1.authenticate.return_value = False
        
        provider2 = MagicMock(spec=AuthProvider)
        provider2.authenticate.return_value = False
        
        auth_manager = AuthManager([provider1, provider2])
        credentials = {"test": "credentials"}
        
        # Act
        result = auth_manager.authenticate(credentials)
        
        # Assert
        assert result is False
        provider1.authenticate.assert_called_once_with(credentials)
        provider2.authenticate.assert_called_once_with(credentials)

    def test_should_authenticate_when_no_providers_exist(self) -> None:
        """Test authentication behavior when no providers exist."""
        # Arrange
        auth_manager = AuthManager([])
        credentials = {"test": "credentials"}
        
        # Act
        result = auth_manager.authenticate(credentials)
        
        # Assert
        assert result is True

    def test_should_get_auth_headers_from_first_provider(self) -> None:
        """Test getting authentication headers from the first provider."""
        # Arrange
        provider1 = MagicMock(spec=AuthProvider)
        provider1.get_auth_headers.return_value = {"X-API-Key": "test-key"}
        
        provider2 = MagicMock(spec=AuthProvider)
        provider2.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        
        auth_manager = AuthManager([provider1, provider2])
        
        # Act
        headers = auth_manager.get_auth_headers()
        
        # Assert
        assert headers == {"X-API-Key": "test-key"}
        provider1.get_auth_headers.assert_called_once()
        provider2.get_auth_headers.assert_not_called()

    def test_should_return_empty_headers_when_no_providers_exist(self) -> None:
        """Test getting authentication headers when no providers exist."""
        # Arrange
        auth_manager = AuthManager([])
        
        # Act
        headers = auth_manager.get_auth_headers()
        
        # Assert
        assert headers == {}

    @patch.dict(os.environ, {"API_KEY_1": "key1", "API_KEY_2": "key2"})
    def test_should_create_from_environment_variables(self) -> None:
        """Test creating an auth manager from environment variables."""
        # Act
        auth_manager = AuthManager.from_env("API_KEY_1", "API_KEY_2")
        
        # Assert
        assert len(auth_manager.providers) == 2
        assert isinstance(auth_manager.providers[0], APIKeyAuth)
        assert isinstance(auth_manager.providers[1], APIKeyAuth)
        assert auth_manager.providers[0].api_key == "key1"
        assert auth_manager.providers[1].api_key == "key2"

    @patch.dict(os.environ, {"GITHUB_TOKEN": "github-token"})
    def test_should_create_from_github_token(self) -> None:
        """Test creating an auth manager with GitHub token."""
        # Act
        auth_manager = AuthManager.from_github_token()
        
        # Assert
        assert len(auth_manager.providers) == 1
        assert isinstance(auth_manager.providers[0], TokenAuth)
        assert auth_manager.providers[0].token == "github-token"
        assert auth_manager.get_auth_headers() == {"Authorization": "Bearer github-token"}


if __name__ == "__main__":
    pytest.main()
