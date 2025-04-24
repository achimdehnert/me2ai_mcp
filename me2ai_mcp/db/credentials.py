"""
Database credential management for ME2AI MCP.

This module provides utilities for managing database credentials
from multiple sources such as environment variables, credential files,
and configuration files.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import os
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class CredentialError(Exception):
    """Base exception for credential-related errors."""
    pass


class DatabaseCredentialManager:
    """Base class for database credential management.
    
    This class provides functionality for loading database credentials
    from multiple sources and in various formats.
    
    Attributes:
        env_prefix: Prefix for environment variables
        credential_file: Path to JSON credentials file
        connection_name: Connection name to use in credentials file
        credentials: Loaded credentials dictionary
        logger: Logger instance
    """
    
    def __init__(
        self,
        env_prefix: str = "DB",
        credential_file: Optional[Union[str, Path]] = None,
        connection_name: Optional[str] = None
    ) -> None:
        """Initialize the credential manager.
        
        Args:
            env_prefix: Prefix for environment variables
            credential_file: Path to JSON credentials file
            connection_name: Connection name to use in credentials file
        """
        self.env_prefix = env_prefix
        
        if credential_file:
            self.credential_file = Path(credential_file)
        else:
            self.credential_file = None
            
        self.connection_name = connection_name or "default"
        
        # Set up logging
        self.logger = logging.getLogger(f"me2ai-mcp-db-credentials")
        
        # Load credentials
        self.credentials = self._load_credentials()
        
        # Log success or failure
        if self._are_credentials_complete(self.credentials):
            self.logger.info(f"Successfully loaded {self.env_prefix} credentials")
        else:
            self.logger.warning(
                f"Incomplete {self.env_prefix} credentials loaded - "
                "connection may fail"
            )
    
    def _load_credentials(self) -> Dict[str, Any]:
        """Load credentials from available sources.
        
        Tries in this order:
        1. Environment variables
        2. Credentials file
        3. Default config files
        
        Returns:
            Dict[str, Any]: Loaded credentials
        """
        # Initialize empty credentials
        credentials = {}
        
        # Try environment variables
        env_credentials = self._load_from_env()
        if self._are_credentials_complete(env_credentials):
            self.logger.info(f"Using {self.env_prefix} credentials from environment variables")
            return env_credentials
        credentials.update(env_credentials)
        
        # Try credentials file
        if self.credential_file:
            file_credentials = self._load_from_file()
            if self._are_credentials_complete(file_credentials):
                self.logger.info(f"Using {self.env_prefix} credentials from file")
                return file_credentials
            credentials.update(file_credentials)
        
        # Try default config
        config_credentials = self._load_from_config()
        if self._are_credentials_complete(config_credentials):
            self.logger.info(f"Using {self.env_prefix} credentials from config file")
            return config_credentials
        credentials.update(config_credentials)
        
        # Try local credential file (in current directory)
        local_credentials = self._load_from_local()
        credentials.update(local_credentials)
        
        return credentials
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load credentials from environment variables.
        
        Returns:
            Dict[str, Any]: Credentials from environment variables
        """
        # This method should be overridden by subclasses
        return {}
    
    def _load_from_file(self) -> Dict[str, Any]:
        """Load credentials from credentials file.
        
        Returns:
            Dict[str, Any]: Credentials from file
        """
        if not self.credential_file:
            return {}
            
        if not self.credential_file.exists():
            self.logger.warning(f"Credentials file not found: {self.credential_file}")
            return {}
            
        try:
            with open(self.credential_file, 'r') as f:
                credentials = json.load(f)
            
            # Get connection by name
            connection = credentials.get(self.connection_name, {})
            
            self.logger.info(f"Loaded credentials from {self.credential_file}")
            return connection
            
        except Exception as e:
            self.logger.warning(f"Could not load credentials from file: {e}")
            return {}
    
    def _load_from_config(self) -> Dict[str, Any]:
        """Load credentials from default config files.
        
        Returns:
            Dict[str, Any]: Credentials from config
        """
        # Try Windsurf config first
        windsurf_config = self._load_from_windsurf_config()
        if self._are_credentials_complete(windsurf_config):
            return windsurf_config
            
        # Try user home directory next
        home_config = self._load_from_home_config()
        if self._are_credentials_complete(home_config):
            return home_config
            
        # Combine results with windsurf taking precedence
        result = {}
        result.update(home_config)
        result.update(windsurf_config)
        
        return result
    
    def _load_from_windsurf_config(self) -> Dict[str, Any]:
        """Load credentials from Windsurf config.
        
        Returns:
            Dict[str, Any]: Credentials from Windsurf config
        """
        config_path = Path.home() / ".codeium" / "windsurf" / "mcp_config.json"
        
        if not config_path.exists():
            return {}
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Get config section based on env_prefix (lowercase)
            section_name = self.env_prefix.lower()
            section = config.get(section_name, {})
            
            if section:
                self.logger.info(f"Found {self.env_prefix} config in Windsurf config")
                
            return section
            
        except Exception as e:
            self.logger.warning(f"Could not load credentials from Windsurf config: {e}")
            return {}
    
    def _load_from_home_config(self) -> Dict[str, Any]:
        """Load credentials from home directory config.
        
        Returns:
            Dict[str, Any]: Credentials from home config
        """
        config_path = Path.home() / f".{self.env_prefix.lower()}_credentials.json"
        
        if not config_path.exists():
            return {}
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Get connection by name
            connection = config.get(self.connection_name, {})
            
            if connection:
                self.logger.info(f"Found {self.env_prefix} credentials in home directory")
                
            return connection
            
        except Exception as e:
            self.logger.warning(f"Could not load credentials from home config: {e}")
            return {}
    
    def _load_from_local(self) -> Dict[str, Any]:
        """Load credentials from local directory.
        
        Returns:
            Dict[str, Any]: Credentials from local directory
        """
        # Look for various common credential file names
        possible_paths = [
            Path(f".{self.env_prefix.lower()}_credentials.json"),
            Path("credentials.json"),
            Path(".env.json"),
            Path(".credentials.json")
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        config = json.load(f)
                    
                    # Get connection by name or use whole file
                    if self.connection_name in config:
                        connection = config.get(self.connection_name, {})
                    else:
                        connection = config
                    
                    if connection:
                        self.logger.info(f"Found {self.env_prefix} credentials in {path}")
                        
                    return connection
                    
                except Exception as e:
                    self.logger.warning(f"Could not load credentials from {path}: {e}")
        
        return {}
    
    def _are_credentials_complete(self, credentials: Dict[str, Any]) -> bool:
        """Check if credentials are complete.
        
        Args:
            credentials: Credentials dictionary
            
        Returns:
            bool: True if credentials are complete
        """
        # This method should be overridden by subclasses
        return False
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for database driver.
        
        Returns:
            Dict[str, Any]: Connection parameters
        """
        # This method should be overridden by subclasses
        return {}


class PostgreSQLCredentialManager(DatabaseCredentialManager):
    """Credential manager for PostgreSQL databases.
    
    This class provides functionality for loading PostgreSQL credentials
    from multiple sources and in various formats.
    """
    
    def __init__(
        self,
        env_prefix: str = "POSTGRES",
        credential_file: Optional[Union[str, Path]] = None,
        connection_name: Optional[str] = None
    ) -> None:
        """Initialize the PostgreSQL credential manager.
        
        Args:
            env_prefix: Prefix for environment variables
            credential_file: Path to JSON credentials file
            connection_name: Connection name to use in credentials file
        """
        super().__init__(
            env_prefix=env_prefix,
            credential_file=credential_file,
            connection_name=connection_name
        )
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load PostgreSQL credentials from environment variables.
        
        Returns:
            Dict[str, Any]: Credentials from environment variables
        """
        # First check for connection string
        uri = os.getenv(f"{self.env_prefix}_URI") or os.getenv(f"{self.env_prefix}_URL")
        
        # Individual parameters
        host = os.getenv(f"{self.env_prefix}_HOST")
        port = os.getenv(f"{self.env_prefix}_PORT", "5432")
        database = os.getenv(f"{self.env_prefix}_DATABASE") or os.getenv(f"{self.env_prefix}_DB")
        username = os.getenv(f"{self.env_prefix}_USER") or os.getenv(f"{self.env_prefix}_USERNAME")
        password = os.getenv(f"{self.env_prefix}_PASSWORD") or os.getenv(f"{self.env_prefix}_PASS")
        
        # Collect credentials
        credentials = {
            "uri": uri,
            "host": host,
            "port": port,
            "database": database,
            "username": username,
            "password": password
        }
        
        # Remove None values
        return {k: v for k, v in credentials.items() if v is not None}
    
    def _are_credentials_complete(self, credentials: Dict[str, Any]) -> bool:
        """Check if PostgreSQL credentials are complete.
        
        Args:
            credentials: Credentials dictionary
            
        Returns:
            bool: True if credentials are complete
        """
        # Check if we have a URI/URL
        if credentials.get("uri") or credentials.get("url"):
            return True
            
        # Check if we have all individual parts
        required = ["host", "database", "username", "password"]
        return all(credentials.get(key) for key in required)
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for psycopg2.
        
        Returns:
            Dict[str, Any]: Connection parameters for psycopg2
        """
        # Check for URI first
        uri = self.credentials.get("uri") or self.credentials.get("url")
        if uri:
            return {"dsn": uri}
        
        # Build connection parameters
        params = {}
        
        # Map credential keys to psycopg2 parameter names
        param_mapping = {
            "host": "host",
            "port": "port",
            "database": "database",
            "username": "user",
            "password": "password"
        }
        
        # Add parameters if they exist
        for cred_key, param_key in param_mapping.items():
            if cred_key in self.credentials:
                value = self.credentials[cred_key]
                
                # Convert port to integer if it's a string
                if cred_key == "port" and isinstance(value, str):
                    try:
                        value = int(value)
                    except ValueError:
                        self.logger.warning(f"Invalid port: {value}, using default 5432")
                        value = 5432
                
                params[param_key] = value
        
        return params
