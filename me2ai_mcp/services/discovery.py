"""
Service discovery for ME2AI MCP.

This module provides service discovery mechanisms for ME2AI MCP
microservices, allowing services to find and communicate with each other.
"""

from typing import Dict, List, Any, Optional, Union, Set
import logging
import asyncio
import json
import time
import socket
import os
from dataclasses import dataclass, field
import uuid
import datetime

from .base import ServiceInfo, ServiceRegistry, get_registry, ServiceStatus

# Configure logging
logger = logging.getLogger("me2ai-mcp-service-discovery")


class ServiceDiscovery:
    """
    Service discovery for ME2AI MCP microservices.
    
    This class provides methods for discovering and monitoring
    services in the ME2AI MCP ecosystem.
    """
    
    def __init__(self, refresh_interval: int = 60):
        """
        Initialize the service discovery.
        
        Args:
            refresh_interval: Interval in seconds to refresh service information
        """
        self.refresh_interval = refresh_interval
        self.registry = get_registry()
        self._refresh_task = None
        self.logger = logging.getLogger("me2ai-mcp-service-discovery")
    
    async def start(self) -> bool:
        """
        Start the service discovery.
        
        Returns:
            bool: True if the service discovery started successfully
        """
        try:
            # Start refresh task
            self._refresh_task = asyncio.create_task(self._refresh_services())
            return True
        except Exception as e:
            self.logger.error(f"Error starting service discovery: {str(e)}")
            return False
    
    async def stop(self) -> bool:
        """
        Stop the service discovery.
        
        Returns:
            bool: True if the service discovery stopped successfully
        """
        try:
            # Stop refresh task
            if self._refresh_task is not None:
                self._refresh_task.cancel()
                self._refresh_task = None
            return True
        except Exception as e:
            self.logger.error(f"Error stopping service discovery: {str(e)}")
            return False
    
    async def _refresh_services(self) -> None:
        """Periodically refresh service information."""
        try:
            while True:
                # Wait for the refresh interval
                await asyncio.sleep(self.refresh_interval)
                
                # Check all registered services
                services = self.registry.list_services()
                for service in services:
                    # Check if service is still alive
                    if await self._check_service_health(service):
                        self.logger.debug(f"Service {service.name} (ID: {service.id}) is healthy")
                    else:
                        self.logger.warning(f"Service {service.name} (ID: {service.id}) is unhealthy")
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            self.logger.info("Service refresh task cancelled")
        except Exception as e:
            # Log error but don't crash
            self.logger.error(f"Error in service refresh: {str(e)}")
    
    async def _check_service_health(self, service: ServiceInfo) -> bool:
        """
        Check if a service is healthy.
        
        Args:
            service: Service to check
            
        Returns:
            bool: True if the service is healthy
        """
        # Check if the service has a recent heartbeat
        current_time = time.time()
        heartbeat_age = current_time - service.last_heartbeat
        
        if heartbeat_age > 180:  # 3 minutes
            self.logger.warning(f"Service {service.name} (ID: {service.id}) has not sent a heartbeat in {heartbeat_age:.1f} seconds")
            return False
        
        # For web services, check the health endpoint
        if "/health" in service.endpoints:
            try:
                # Try to connect to the service
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2.0)
                    result = s.connect_ex((service.host, service.port))
                    if result != 0:
                        self.logger.warning(f"Could not connect to service {service.name} (ID: {service.id}) at {service.host}:{service.port}")
                        return False
                
                # TODO: Make an HTTP request to the health endpoint
                # This would use aiohttp or similar to make the actual HTTP request
                
                # For now, assume the service is healthy if we can connect
                return True
                
            except Exception as e:
                self.logger.warning(f"Error checking service health: {str(e)}")
                return False
        
        # Default to healthy if no health check is available
        return True
    
    def find_service(self, name: str) -> Optional[ServiceInfo]:
        """
        Find a service by name.
        
        Args:
            name: Name of the service to find
            
        Returns:
            ServiceInfo or None: The service if found
        """
        return self.registry.get_service(name)
    
    def find_services_by_endpoint(self, endpoint: str, method: str = "GET") -> List[ServiceInfo]:
        """
        Find services that provide a specific endpoint.
        
        Args:
            endpoint: Endpoint path to look for
            method: HTTP method to match
            
        Returns:
            List[ServiceInfo]: List of services that provide the endpoint
        """
        services = []
        
        for service in self.registry.list_services():
            # Check if the service provides the endpoint
            endpoint_key = f"{method.upper()}{endpoint}"
            if endpoint in service.endpoints and service.endpoints[endpoint] == method.upper():
                services.append(service)
                
        return services
    
    def list_services(self) -> List[ServiceInfo]:
        """
        List all available services.
        
        Returns:
            List[ServiceInfo]: List of all available services
        """
        return self.registry.list_services()
    
    def get_service_status(self, service_id: str) -> Optional[ServiceStatus]:
        """
        Get the status of a service.
        
        Args:
            service_id: ID of the service
            
        Returns:
            ServiceStatus or None: The status of the service if found
        """
        service = self.registry.get_service_by_id(service_id)
        if service is not None:
            return service.status
        return None
