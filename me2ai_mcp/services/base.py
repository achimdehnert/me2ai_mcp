"""
Base service architecture for ME2AI MCP.

This module provides the core service classes and interfaces
for building modular, scalable microservices within ME2AI MCP.
"""

from typing import Dict, List, Any, Optional, Union, Type, Callable, Awaitable, Set
import logging
import asyncio
import json
import time
import uuid
import os
from dataclasses import dataclass, field
from enum import Enum
import socket
import sys
import contextlib
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger("me2ai-mcp-services")


class ServiceStatus(Enum):
    """Status of a service instance."""
    
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ServiceInfo:
    """Information about a registered service."""
    
    id: str
    name: str
    host: str
    port: int
    status: ServiceStatus
    version: str
    endpoints: Dict[str, str]
    metadata: Dict[str, Any]
    last_heartbeat: float = field(default_factory=time.time)
    registration_time: float = field(default_factory=time.time)


@dataclass
class ServiceEndpoint:
    """Definition of a service endpoint."""
    
    path: str
    method: str
    description: str
    parameters: Optional[Dict[str, Any]] = None
    response_schema: Optional[Dict[str, Any]] = None
    auth_required: bool = False
    rate_limit: Optional[int] = None


class ServiceRegistry:
    """Registry for discovering and managing services."""
    
    def __init__(self):
        """Initialize the service registry."""
        self.services: Dict[str, ServiceInfo] = {}
        self.logger = logging.getLogger("me2ai-mcp-service-registry")
        self._cleanup_task = None
    
    def register(self, service_info: ServiceInfo) -> bool:
        """
        Register a service with the registry.
        
        Args:
            service_info: Information about the service to register
            
        Returns:
            bool: True if registration was successful
        """
        self.services[service_info.id] = service_info
        self.logger.info(f"Registered service: {service_info.name} (ID: {service_info.id})")
        
        # Start cleanup task if not already running
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_stale_services())
            
        return True
    
    def unregister(self, service_id: str) -> bool:
        """
        Unregister a service from the registry.
        
        Args:
            service_id: ID of the service to unregister
            
        Returns:
            bool: True if unregistration was successful
        """
        if service_id in self.services:
            service_info = self.services.pop(service_id)
            self.logger.info(f"Unregistered service: {service_info.name} (ID: {service_id})")
            return True
        return False
    
    def get_service(self, service_name: str) -> Optional[ServiceInfo]:
        """
        Get information about a service by name.
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            ServiceInfo or None: Information about the service if found
        """
        for service_id, service_info in self.services.items():
            if service_info.name == service_name and service_info.status == ServiceStatus.RUNNING:
                return service_info
        return None
    
    def get_service_by_id(self, service_id: str) -> Optional[ServiceInfo]:
        """
        Get information about a service by ID.
        
        Args:
            service_id: ID of the service to retrieve
            
        Returns:
            ServiceInfo or None: Information about the service if found
        """
        return self.services.get(service_id)
    
    def list_services(self) -> List[ServiceInfo]:
        """
        List all registered services.
        
        Returns:
            List[ServiceInfo]: List of all registered services
        """
        return list(self.services.values())
    
    def heartbeat(self, service_id: str) -> bool:
        """
        Update the heartbeat timestamp for a service.
        
        Args:
            service_id: ID of the service to update
            
        Returns:
            bool: True if heartbeat was updated successfully
        """
        if service_id in self.services:
            self.services[service_id].last_heartbeat = time.time()
            return True
        return False
    
    async def _cleanup_stale_services(self) -> None:
        """Periodically clean up stale service registrations."""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = time.time()
                stale_threshold = current_time - 180  # 3 minutes
                
                # Find stale services
                stale_services = [
                    service_id for service_id, info in self.services.items()
                    if info.last_heartbeat < stale_threshold
                ]
                
                # Unregister stale services
                for service_id in stale_services:
                    self.logger.warning(f"Removing stale service: {self.services[service_id].name} (ID: {service_id})")
                    self.unregister(service_id)
                    
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            self.logger.info("Service cleanup task cancelled")
        except Exception as e:
            # Log error but don't crash
            self.logger.error(f"Error in service cleanup: {str(e)}")


# Singleton registry instance
_registry = None


def get_registry() -> ServiceRegistry:
    """
    Get the global service registry instance.
    
    Returns:
        ServiceRegistry: The global service registry
    """
    global _registry
    if _registry is None:
        _registry = ServiceRegistry()
    return _registry


class BaseService:
    """
    Base class for all ME2AI MCP microservices.
    
    This class provides the foundation for building modular,
    scalable services with standard health checks, service
    discovery, and lifecycle management.
    """
    
    def __init__(
        self,
        name: str,
        host: str = "localhost",
        port: int = 0,  # 0 means auto-assign available port
        version: str = "0.1.0",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the service.
        
        Args:
            name: Service name
            host: Host to bind the service to
            port: Port to bind the service to (0 for auto-assign)
            version: Service version
            metadata: Additional service metadata
        """
        self.name = name
        self.host = host
        self.port = port
        self.version = version
        self.metadata = metadata or {}
        
        # Generate unique service ID
        self.id = f"{name}-{str(uuid.uuid4())[:8]}"
        
        # Initialize service state
        self.status = ServiceStatus.INITIALIZING
        self.start_time = None
        self.endpoints: Dict[str, ServiceEndpoint] = {}
        
        # Set up logging
        self.logger = logging.getLogger(f"me2ai-mcp-service-{name}")
        
        # Register built-in endpoints
        self._register_builtin_endpoints()
    
    def _register_builtin_endpoints(self) -> None:
        """Register the built-in service endpoints."""
        self.register_endpoint(
            path="/health",
            method="GET",
            description="Health check endpoint",
            parameters={},
            response_schema={
                "status": "string",
                "uptime": "number",
                "version": "string"
            }
        )
        
        self.register_endpoint(
            path="/info",
            method="GET",
            description="Service information endpoint",
            parameters={},
            response_schema={
                "id": "string",
                "name": "string",
                "version": "string",
                "status": "string",
                "endpoints": "object",
                "metadata": "object"
            }
        )
    
    def register_endpoint(
        self,
        path: str,
        method: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        auth_required: bool = False,
        rate_limit: Optional[int] = None
    ) -> None:
        """
        Register a service endpoint.
        
        Args:
            path: URL path for the endpoint
            method: HTTP method (GET, POST, etc.)
            description: Description of the endpoint
            parameters: Parameter schema for the endpoint
            response_schema: Response schema for the endpoint
            auth_required: Whether authentication is required
            rate_limit: Rate limit for the endpoint in requests per minute
        """
        endpoint_key = f"{method.upper()}{path}"
        self.endpoints[endpoint_key] = ServiceEndpoint(
            path=path,
            method=method.upper(),
            description=description,
            parameters=parameters,
            response_schema=response_schema,
            auth_required=auth_required,
            rate_limit=rate_limit
        )
        self.logger.debug(f"Registered endpoint: {method.upper()} {path}")
    
    async def start(self) -> bool:
        """
        Start the service.
        
        Returns:
            bool: True if the service started successfully
        """
        self.status = ServiceStatus.STARTING
        
        try:
            # Find available port if auto-assign
            if self.port == 0:
                self.port = self._find_available_port()
                
            # Register with service registry
            registry = get_registry()
            registration_result = registry.register(
                ServiceInfo(
                    id=self.id,
                    name=self.name,
                    host=self.host,
                    port=self.port,
                    status=ServiceStatus.STARTING,
                    version=self.version,
                    endpoints={endpoint.path: endpoint.method for endpoint in self.endpoints.values()},
                    metadata=self.metadata
                )
            )
            
            if not registration_result:
                self.logger.error(f"Failed to register service {self.name}")
                self.status = ServiceStatus.ERROR
                return False
                
            # Start heartbeat task
            self._heartbeat_task = asyncio.create_task(self._send_heartbeat())
            
            # Service is now running
            self.status = ServiceStatus.RUNNING
            self.start_time = time.time()
            
            self.logger.info(f"Service {self.name} started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting service: {str(e)}")
            self.status = ServiceStatus.ERROR
            return False
    
    async def stop(self) -> bool:
        """
        Stop the service.
        
        Returns:
            bool: True if the service stopped successfully
        """
        self.status = ServiceStatus.STOPPING
        
        try:
            # Stop heartbeat task
            if hasattr(self, "_heartbeat_task"):
                self._heartbeat_task.cancel()
                
            # Unregister from service registry
            registry = get_registry()
            registry.unregister(self.id)
            
            # Service is now stopped
            self.status = ServiceStatus.STOPPED
            
            self.logger.info(f"Service {self.name} stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping service: {str(e)}")
            self.status = ServiceStatus.ERROR
            return False
    
    async def _send_heartbeat(self) -> None:
        """Periodically send heartbeat to the service registry."""
        try:
            registry = get_registry()
            
            while self.status == ServiceStatus.RUNNING:
                registry.heartbeat(self.id)
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
        except asyncio.CancelledError:
            # Task was cancelled, this is expected during shutdown
            pass
        except Exception as e:
            # Log error but don't crash
            self.logger.error(f"Error sending heartbeat: {str(e)}")
    
    def _find_available_port(self) -> int:
        """
        Find an available port to bind the service to.
        
        Returns:
            int: Available port number
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check for the service.
        
        Returns:
            Dict[str, Any]: Health check information
        """
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            "status": self.status.value,
            "uptime": uptime,
            "version": self.version
        }
    
    async def get_info(self) -> Dict[str, Any]:
        """
        Get information about the service.
        
        Returns:
            Dict[str, Any]: Service information
        """
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "host": self.host,
            "port": self.port,
            "endpoints": {
                endpoint.path: {
                    "method": endpoint.method,
                    "description": endpoint.description,
                    "auth_required": endpoint.auth_required
                }
                for endpoint in self.endpoints.values()
            },
            "metadata": self.metadata
        }


@dataclass
class ServiceClient:
    """
    Client for interacting with ME2AI MCP services.
    
    This class provides a high-level interface for discovering and
    communicating with ME2AI MCP services.
    """
    
    service_name: str
    timeout: float = 30.0
    auto_reconnect: bool = True
    _service_info: Optional[ServiceInfo] = field(default=None, init=False)
    
    async def connect(self) -> bool:
        """
        Connect to the service.
        
        Returns:
            bool: True if connection was successful
        """
        registry = get_registry()
        self._service_info = registry.get_service(self.service_name)
        
        if self._service_info is None:
            return False
            
        return True
    
    async def call(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call a service endpoint.
        
        Args:
            endpoint: Endpoint path to call
            method: HTTP method to use
            params: Parameters to send with the request
            
        Returns:
            Dict[str, Any]: Response from the service
            
        Raises:
            ValueError: If not connected to the service
            RuntimeError: If the service call fails
        """
        if self._service_info is None:
            if self.auto_reconnect:
                if not await self.connect():
                    raise ValueError(f"Could not connect to service: {self.service_name}")
            else:
                raise ValueError(f"Not connected to service: {self.service_name}")
        
        # TODO: Implement HTTP client to call the service
        # This would use aiohttp or similar to make the actual HTTP request
        
        # For now, just return a placeholder response
        return {
            "success": True,
            "message": f"Called {method} {endpoint} on {self.service_name}",
            "params": params
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the service is healthy.
        
        Returns:
            Dict[str, Any]: Health check response
            
        Raises:
            ValueError: If not connected to the service
        """
        return await self.call("/health", "GET")
    
    async def get_info(self) -> Dict[str, Any]:
        """
        Get information about the service.
        
        Returns:
            Dict[str, Any]: Service information
            
        Raises:
            ValueError: If not connected to the service
        """
        return await self.call("/info", "GET")


class ServiceManager:
    """
    Manager for controlling multiple ME2AI MCP services.
    
    This class provides a high-level interface for starting,
    stopping, and monitoring multiple services.
    """
    
    def __init__(self):
        """Initialize the service manager."""
        self.services: Dict[str, BaseService] = {}
        self.logger = logging.getLogger("me2ai-mcp-service-manager")
    
    def register_service(self, service: BaseService) -> bool:
        """
        Register a service with the manager.
        
        Args:
            service: Service to register
            
        Returns:
            bool: True if registration was successful
        """
        if service.id in self.services:
            self.logger.warning(f"Service already registered: {service.name} (ID: {service.id})")
            return False
            
        self.services[service.id] = service
        self.logger.info(f"Service registered with manager: {service.name} (ID: {service.id})")
        return True
    
    def unregister_service(self, service_id: str) -> bool:
        """
        Unregister a service from the manager.
        
        Args:
            service_id: ID of the service to unregister
            
        Returns:
            bool: True if unregistration was successful
        """
        if service_id in self.services:
            service = self.services.pop(service_id)
            self.logger.info(f"Service unregistered from manager: {service.name} (ID: {service_id})")
            return True
        return False
    
    async def start_all_services(self) -> Dict[str, bool]:
        """
        Start all registered services.
        
        Returns:
            Dict[str, bool]: Map of service IDs to start results
        """
        results = {}
        
        for service_id, service in self.services.items():
            results[service_id] = await service.start()
            
        return results
    
    async def stop_all_services(self) -> Dict[str, bool]:
        """
        Stop all registered services.
        
        Returns:
            Dict[str, bool]: Map of service IDs to stop results
        """
        results = {}
        
        for service_id, service in self.services.items():
            results[service_id] = await service.stop()
            
        return results
    
    async def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health information for all services.
        
        Returns:
            Dict[str, Dict[str, Any]]: Map of service IDs to health info
        """
        results = {}
        
        for service_id, service in self.services.items():
            try:
                results[service_id] = await service.health_check()
            except Exception as e:
                results[service_id] = {"status": "error", "message": str(e)}
                
        return results
    
    def get_service(self, service_id: str) -> Optional[BaseService]:
        """
        Get a service by ID.
        
        Args:
            service_id: ID of the service to retrieve
            
        Returns:
            BaseService or None: The service if found
        """
        return self.services.get(service_id)
    
    def get_service_by_name(self, service_name: str) -> Optional[BaseService]:
        """
        Get a service by name.
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            BaseService or None: The service if found
        """
        for service in self.services.values():
            if service.name == service_name:
                return service
        return None
