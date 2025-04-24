"""
Tests for the database credential management system.

This module contains tests for the credential management functionality
in the ME2AI MCP package.
"""

import os
import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch

from me2ai_mcp.db.credentials import (
    DatabaseCredentials,
    CredentialSourceType,
    get_credentials
)


class TestDatabaseCredentials:
    """Tests for the DatabaseCredentials class."""
    
    def test_should_create_valid_credentials(self):
        """Test that valid credentials can be created."""
        # Arrange
        host = "localhost"
        port = 5432
        username = "testuser"
        password = "testpass"
        
        # Act
        credentials = DatabaseCredentials(
            host=host,
            port=port,
            username=username,
            password=password,
            source_type=CredentialSourceType.ENV
        )
        
        # Assert
        assert credentials.host == host
        assert credentials.port == port
        assert credentials.username == username
        assert credentials.password == password
        assert credentials.source_type == CredentialSourceType.ENV
    
    def test_should_validate_port_range(self):
        """Test that port is validated correctly."""
        # Arrange, Act, Assert
        with pytest.raises(ValueError):
            DatabaseCredentials(
                host="localhost",
                port=0,  # Invalid port
                username="testuser",
                password="testpass"
            )
        
        with pytest.raises(ValueError):
            DatabaseCredentials(
                host="localhost",
                port=65536,  # Invalid port
                username="testuser",
                password="testpass"
            )
    
    def test_should_validate_required_fields(self):
        """Test that required fields are validated correctly."""
        # Arrange, Act, Assert
        with pytest.raises(ValueError):
            DatabaseCredentials(
                host="",  # Empty host
                port=5432,
                username="testuser",
                password="testpass"
            )
        
        with pytest.raises(ValueError):
            DatabaseCredentials(
                host="localhost",
                port=5432,
                username="",  # Empty username
                password="testpass"
            )


class TestGetCredentials:
    """Tests for the get_credentials function."""
    
    @patch.dict(os.environ, {
        "TEST_HOST": "test-host",
        "TEST_PORT": "5432",
        "TEST_USERNAME": "test-username",
        "TEST_PASSWORD": "test-password"
    })
    def test_should_get_credentials_from_env(self):
        """Test that credentials can be retrieved from environment variables."""
        # Arrange
        env_prefix = "TEST"
        
        # Act
        credentials = get_credentials(env_prefix=env_prefix)
        
        # Assert
        assert credentials.host == "test-host"
        assert credentials.port == 5432
        assert credentials.username == "test-username"
        assert credentials.password == "test-password"
        assert credentials.source_type == CredentialSourceType.ENV
    
    def test_should_get_credentials_from_file(self):
        """Test that credentials can be retrieved from a JSON file."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump({
                "database": {
                    "host": "file-host",
                    "port": 5432,
                    "username": "file-username",
                    "password": "file-password"
                }
            }, f)
            credential_file = Path(f.name)
        
        # Act
        try:
            credentials = get_credentials(credential_file=credential_file)
            
            # Assert
            assert credentials.host == "file-host"
            assert credentials.port == 5432
            assert credentials.username == "file-username"
            assert credentials.password == "file-password"
            assert credentials.source_type == CredentialSourceType.FILE
        
        finally:
            # Clean up
            if credential_file.exists():
                credential_file.unlink()
    
    def test_should_get_credentials_from_named_connection(self):
        """Test that credentials can be retrieved from a named connection."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump({
                "connections": {
                    "test-connection": {
                        "host": "named-host",
                        "port": 5432,
                        "username": "named-username",
                        "password": "named-password"
                    }
                }
            }, f)
            credential_file = Path(f.name)
        
        # Act
        try:
            credentials = get_credentials(
                credential_file=credential_file,
                connection_name="test-connection"
            )
            
            # Assert
            assert credentials.host == "named-host"
            assert credentials.port == 5432
            assert credentials.username == "named-username"
            assert credentials.password == "named-password"
            assert credentials.source_type == CredentialSourceType.FILE
        
        finally:
            # Clean up
            if credential_file.exists():
                credential_file.unlink()
    
    @patch.dict(os.environ, {
        "TEST_HOST": "env-host",
        "TEST_PORT": "5432",
        "TEST_USERNAME": "env-username",
        "TEST_PASSWORD": "env-password"
    })
    def test_should_prioritize_file_over_env(self):
        """Test that file credentials take priority over environment variables."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump({
                "database": {
                    "host": "file-host",
                    "port": 5432,
                    "username": "file-username",
                    "password": "file-password"
                }
            }, f)
            credential_file = Path(f.name)
        
        # Act
        try:
            credentials = get_credentials(
                env_prefix="TEST",
                credential_file=credential_file
            )
            
            # Assert
            assert credentials.host == "file-host"  # File takes priority
            assert credentials.port == 5432
            assert credentials.username == "file-username"
            assert credentials.password == "file-password"
            assert credentials.source_type == CredentialSourceType.FILE
        
        finally:
            # Clean up
            if credential_file.exists():
                credential_file.unlink()
    
    def test_should_raise_error_for_missing_credentials(self):
        """Test that an error is raised when credentials are missing."""
        # Arrange, Act, Assert
        with pytest.raises(ValueError):
            get_credentials(
                env_prefix="NONEXISTENT",
                credential_file=None
            )
        
        # Test with non-existent file
        with pytest.raises(ValueError):
            get_credentials(
                credential_file=Path("/nonexistent/file.json")
            )
    
    def test_should_raise_error_for_invalid_connection_name(self):
        """Test that an error is raised for an invalid connection name."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump({
                "connections": {
                    "connection1": {
                        "host": "host",
                        "port": 5432,
                        "username": "username",
                        "password": "password"
                    }
                }
            }, f)
            credential_file = Path(f.name)
        
        # Act, Assert
        try:
            with pytest.raises(ValueError):
                get_credentials(
                    credential_file=credential_file,
                    connection_name="nonexistent-connection"
                )
        
        finally:
            # Clean up
            if credential_file.exists():
                credential_file.unlink()
