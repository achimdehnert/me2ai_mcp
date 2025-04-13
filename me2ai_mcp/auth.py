"""
Authentication utilities for ME2AI MCP servers.

This module provides authentication managers and methods for securing
MCP server endpoints and API access.
"""
from typing import Dict, List, Any, Optional, Union, Callable, Protocol
import os
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger("me2ai-mcp-auth")


class AuthProvider(ABC):
    """Abstract base class for authentication providers."""
    
    @abstractmethod
    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate a request using the provided credentials.
        
        Args:
            credentials: Authentication credentials
            
        Returns:
            Whether authentication was successful
        """
        pass
    
    @abstractmethod
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for outgoing requests.
        
        Returns:
            Dictionary of authentication headers
        """
        pass


class APIKeyAuth(AuthProvider):
    """API key-based authentication for MCP servers."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        env_var_name: Optional[str] = None,
        header_name: str = "X-API-Key"
    ) -> None:
        """Initialize API key authentication.
        
        Args:
            api_key: API key (optional if env_var_name is provided)
            env_var_name: Name of environment variable containing API key
            header_name: Name of header for API key
        """
        self.header_name = header_name
        
        # Load from environment if not provided directly
        if api_key is None and env_var_name:
            load_dotenv()
            api_key = os.getenv(env_var_name)
            
        self.api_key = api_key
        
        if not self.api_key:
            logger.warning(f"No API key provided for {self.__class__.__name__}")
    
    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate using API key.
        
        Args:
            credentials: Dictionary containing the API key
            
        Returns:
            Whether authentication was successful
        """
        if not self.api_key:
            # If no API key is configured, authentication is disabled
            return True
            
        # Extract API key from various potential sources
        request_api_key = credentials.get(self.header_name, credentials.get("api_key"))
        
        if not request_api_key:
            logger.warning("Authentication failed: No API key provided in request")
            return False
            
        # Compare API keys
        return request_api_key == self.api_key
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get API key authentication headers.
        
        Returns:
            Dictionary containing API key header
        """
        if not self.api_key:
            return {}
            
        return {self.header_name: self.api_key}


class TokenAuth(AuthProvider):
    """Token-based authentication for MCP servers."""
    
    def __init__(
        self, 
        token: Optional[str] = None,
        env_var_name: Optional[str] = None,
        auth_scheme: str = "Bearer"
    ) -> None:
        """Initialize token authentication.
        
        Args:
            token: Authentication token (optional if env_var_name is provided)
            env_var_name: Name of environment variable containing token
            auth_scheme: Authentication scheme (e.g., "Bearer")
        """
        self.auth_scheme = auth_scheme
        
        # Load from environment if not provided directly
        if token is None and env_var_name:
            load_dotenv()
            token = os.getenv(env_var_name)
            
        self.token = token
        
        if not self.token:
            logger.warning(f"No token provided for {self.__class__.__name__}")
    
    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate using token.
        
        Args:
            credentials: Dictionary containing the authorization header
            
        Returns:
            Whether authentication was successful
        """
        if not self.token:
            # If no token is configured, authentication is disabled
            return True
            
        # Extract token from authorization header
        auth_header = credentials.get("Authorization", "")
        
        if not auth_header.startswith(f"{self.auth_scheme} "):
            logger.warning(f"Authentication failed: Invalid Authorization header format (expected {self.auth_scheme})")
            return False
            
        # Extract the token part
        request_token = auth_header[len(f"{self.auth_scheme} "):]
        
        # Compare tokens
        return request_token == self.token
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get token authentication headers.
        
        Returns:
            Dictionary containing Authorization header
        """
        if not self.token:
            return {}
            
        return {"Authorization": f"{self.auth_scheme} {self.token}"}


class AuthManager:
    """Authentication manager for ME2AI MCP servers."""
    
    def __init__(self, providers: Optional[List[AuthProvider]] = None) -> None:
        """Initialize the authentication manager.
        
        Args:
            providers: List of authentication providers
        """
        self.providers = providers or []
        self.logger = logging.getLogger("me2ai-mcp-auth-manager")
    
    def add_provider(self, provider: AuthProvider) -> None:
        """Add an authentication provider.
        
        Args:
            provider: Authentication provider to add
        """
        self.providers.append(provider)
        
    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate a request using all providers.
        
        Authentication succeeds if ANY provider authenticates successfully.
        If no providers are configured, authentication is always successful.
        
        Args:
            credentials: Authentication credentials
            
        Returns:
            Whether authentication was successful
        """
        if not self.providers:
            # If no providers are configured, authentication is disabled
            return True
            
        # Try each provider
        for provider in self.providers:
            if provider.authenticate(credentials):
                return True
                
        self.logger.warning("Authentication failed: No provider authenticated the request")
        return False
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers from the first provider.
        
        Returns:
            Dictionary of authentication headers
        """
        if not self.providers:
            return {}
            
        # Use the first provider's headers
        return self.providers[0].get_auth_headers()
    
    @classmethod
    def from_env(cls, *env_var_names: str) -> "AuthManager":
        """Create an authentication manager from environment variables.
        
        This method creates API key authentication providers for each
        environment variable name provided.
        
        Args:
            env_var_names: Names of environment variables containing API keys
            
        Returns:
            Configured authentication manager
        """
        load_dotenv()
        
        providers = []
        
        for env_var_name in env_var_names:
            if os.getenv(env_var_name):
                providers.append(APIKeyAuth(env_var_name=env_var_name))
                
        return cls(providers)
        
    @classmethod
    def from_github_token(cls) -> "AuthManager":
        """Create an authentication manager using a GitHub token.
        
        This method checks the following environment variables in order:
        - GITHUB_API_KEY
        - GITHUB_TOKEN
        - GITHUB_ACCESS_TOKEN
        
        Returns:
            Configured authentication manager with GitHub token
        """
        load_dotenv()
        
        # Try different potential environment variable names
        token = (
            os.getenv("GITHUB_API_KEY") or 
            os.getenv("GITHUB_TOKEN") or 
            os.getenv("GITHUB_ACCESS_TOKEN")
        )
        
        if token:
            return cls([TokenAuth(token=token, auth_scheme="Bearer")])
        else:
            logger.warning("No GitHub token found in environment variables")
            return cls()
