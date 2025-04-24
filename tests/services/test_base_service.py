"""
Tests for ME2AI MCP service architecture.

This module contains tests for the base service architecture
components of ME2AI MCP.
"""

import pytest
import unittest.mock as mock
import asyncio
import time
from typing import Dict, Any

from me2ai_mcp.services.base import (
    BaseService,
    ServiceRegistry,
    ServiceStatus,
    ServiceInfo,
    get_registry
)


class TestServiceRegistry:
    """Tests for the ServiceRegistry class."""
    
    def test_should_register_service(self):
        """Test that services can be registered."""
        registry = ServiceRegistry()
        
        # Create a mock service info
        service_info = ServiceInfo(
            id="test-service-1",
            name="TestService",
            host="localhost",
            port=8000,
            status=ServiceStatus.RUNNING,
            version="0.1.0",
            endpoints={"/health": "GET"},
            metadata={"test": True}
        )
        
        # Register the service
        result = registry.register(service_info)
        
        # Verify the result
        assert result is True
        assert "test-service-1" in registry.services
        assert registry.services["test-service-1"] == service_info
        
    def test_should_unregister_service(self):
        """Test that services can be unregistered."""
        registry = ServiceRegistry()
        
        # Create a mock service info
        service_info = ServiceInfo(
            id="test-service-2",
            name="TestService",
            host="localhost",
            port=8000,
            status=ServiceStatus.RUNNING,
            version="0.1.0",
            endpoints={"/health": "GET"},
            metadata={"test": True}
        )
        
        # Register the service
        registry.register(service_info)
        
        # Unregister the service
        result = registry.unregister("test-service-2")
        
        # Verify the result
        assert result is True
        assert "test-service-2" not in registry.services
        
    def test_should_get_service_by_name(self):
        """Test that services can be retrieved by name."""
        registry = ServiceRegistry()
        
        # Create a mock service info
        service_info = ServiceInfo(
            id="test-service-3",
            name="UniqueServiceName",
            host="localhost",
            port=8000,
            status=ServiceStatus.RUNNING,
            version="0.1.0",
            endpoints={"/health": "GET"},
            metadata={"test": True}
        )
        
        # Register the service
        registry.register(service_info)
        
        # Get the service by name
        result = registry.get_service("UniqueServiceName")
        
        # Verify the result
        assert result is not None
        assert result.id == "test-service-3"
        assert result.name == "UniqueServiceName"
        
    def test_should_get_service_by_id(self):
        """Test that services can be retrieved by ID."""
        registry = ServiceRegistry()
        
        # Create a mock service info
        service_info = ServiceInfo(
            id="test-service-4",
            name="TestService",
            host="localhost",
            port=8000,
            status=ServiceStatus.RUNNING,
            version="0.1.0",
            endpoints={"/health": "GET"},
            metadata={"test": True}
        )
        
        # Register the service
        registry.register(service_info)
        
        # Get the service by ID
        result = registry.get_service_by_id("test-service-4")
        
        # Verify the result
        assert result is not None
        assert result.id == "test-service-4"
        assert result.name == "TestService"
        
    def test_should_update_heartbeat(self):
        """Test that service heartbeats can be updated."""
        registry = ServiceRegistry()
        
        # Create a mock service info
        service_info = ServiceInfo(
            id="test-service-5",
            name="TestService",
            host="localhost",
            port=8000,
            status=ServiceStatus.RUNNING,
            version="0.1.0",
            endpoints={"/health": "GET"},
            metadata={"test": True}
        )
        
        # Register the service
        registry.register(service_info)
        
        # Get the initial heartbeat
        initial_heartbeat = registry.services["test-service-5"].last_heartbeat
        
        # Wait a moment
        time.sleep(0.1)
        
        # Update the heartbeat
        result = registry.heartbeat("test-service-5")
        
        # Verify the result
        assert result is True
        assert registry.services["test-service-5"].last_heartbeat > initial_heartbeat
        
    def test_should_list_services(self):
        """Test that all services can be listed."""
        registry = ServiceRegistry()
        
        # Create mock service infos
        service_info_1 = ServiceInfo(
            id="test-service-6",
            name="TestService1",
            host="localhost",
            port=8000,
            status=ServiceStatus.RUNNING,
            version="0.1.0",
            endpoints={"/health": "GET"},
            metadata={"test": True}
        )
        
        service_info_2 = ServiceInfo(
            id="test-service-7",
            name="TestService2",
            host="localhost",
            port=8001,
            status=ServiceStatus.RUNNING,
            version="0.1.0",
            endpoints={"/health": "GET"},
            metadata={"test": True}
        )
        
        # Register the services
        registry.register(service_info_1)
        registry.register(service_info_2)
        
        # List the services
        result = registry.list_services()
        
        # Verify the result
        assert len(result) == 2
        assert service_info_1 in result
        assert service_info_2 in result
        
    def test_should_get_registry_singleton(self):
        """Test that get_registry returns a singleton instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        
        # Verify that the same instance is returned
        assert registry1 is registry2


@pytest.mark.asyncio
class TestBaseService:
    """Tests for the BaseService class."""
    
    async def test_should_initialize_service(self):
        """Test that services can be initialized."""
        service = BaseService(
            name="TestService",
            host="localhost",
            port=8000,
            version="0.1.0",
            metadata={"test": True}
        )
        
        # Verify service properties
        assert service.name == "TestService"
        assert service.host == "localhost"
        assert service.port == 8000
        assert service.version == "0.1.0"
        assert service.metadata == {"test": True}
        assert service.status == ServiceStatus.INITIALIZING
        
    async def test_should_register_endpoint(self):
        """Test that endpoints can be registered."""
        service = BaseService(
            name="TestService",
            host="localhost",
            port=8000
        )
        
        # Register an endpoint
        service.register_endpoint(
            path="/test",
            method="GET",
            description="Test endpoint",
            parameters={"param": {"type": "string"}},
            response_schema={"result": "string"},
            auth_required=True,
            rate_limit=100
        )
        
        # Verify the endpoint
        assert "GET/test" in service.endpoints
        endpoint = service.endpoints["GET/test"]
        assert endpoint.path == "/test"
        assert endpoint.method == "GET"
        assert endpoint.description == "Test endpoint"
        assert endpoint.parameters == {"param": {"type": "string"}}
        assert endpoint.response_schema == {"result": "string"}
        assert endpoint.auth_required is True
        assert endpoint.rate_limit == 100
        
    @mock.patch("me2ai_mcp.services.base.get_registry")
    async def test_should_start_service(self, mock_get_registry):
        """Test that services can be started."""
        # Mock service registry
        registry = mock.MagicMock()
        registry.register.return_value = True
        mock_get_registry.return_value = registry
        
        # Create service
        service = BaseService(
            name="TestService",
            host="localhost",
            port=8000
        )
        
        # Mock the _ensure_server_running method
        service._ensure_server_running = mock.AsyncMock(return_value=True)
        
        # Start the service
        with mock.patch.object(service, "_send_heartbeat", return_value=None):
            service._send_heartbeat = mock.AsyncMock()
            result = await service.start()
        
        # Verify the result
        assert result is True
        assert service.status == ServiceStatus.RUNNING
        assert service.start_time is not None
        
        # Verify registry interactions
        registry.register.assert_called_once()
        
    @mock.patch("me2ai_mcp.services.base.get_registry")
    async def test_should_stop_service(self, mock_get_registry):
        """Test that services can be stopped."""
        # Mock service registry
        registry = mock.MagicMock()
        registry.unregister.return_value = True
        mock_get_registry.return_value = registry
        
        # Create service
        service = BaseService(
            name="TestService",
            host="localhost",
            port=8000
        )
        
        # Set service state
        service.status = ServiceStatus.RUNNING
        service.start_time = time.time()
        
        # Add heartbeat task mock
        service._heartbeat_task = mock.MagicMock()
        service._heartbeat_task.cancel = mock.MagicMock()
        
        # Stop the service
        result = await service.stop()
        
        # Verify the result
        assert result is True
        assert service.status == ServiceStatus.STOPPED
        
        # Verify heartbeat task cancellation
        service._heartbeat_task.cancel.assert_called_once()
        
        # Verify registry interactions
        registry.unregister.assert_called_once_with(service.id)
        
    async def test_should_get_health_check(self):
        """Test that health checks return the expected format."""
        service = BaseService(
            name="TestService",
            host="localhost",
            port=8000,
            version="0.1.0"
        )
        
        # Set service state
        service.status = ServiceStatus.RUNNING
        service.start_time = time.time() - 60  # Started 60 seconds ago
        
        # Get health check
        health = await service.health_check()
        
        # Verify health check format
        assert health["status"] == "running"
        assert health["version"] == "0.1.0"
        assert health["uptime"] >= 60
        
    async def test_should_get_service_info(self):
        """Test that service info returns the expected format."""
        service = BaseService(
            name="TestService",
            host="localhost",
            port=8000,
            version="0.1.0",
            metadata={"test": True}
        )
        
        # Set service state
        service.status = ServiceStatus.RUNNING
        
        # Register an endpoint
        service.register_endpoint(
            path="/test",
            method="GET",
            description="Test endpoint"
        )
        
        # Get service info
        info = await service.get_info()
        
        # Verify service info format
        assert info["id"] == service.id
        assert info["name"] == "TestService"
        assert info["version"] == "0.1.0"
        assert info["status"] == "running"
        assert info["host"] == "localhost"
        assert info["port"] == 8000
        assert info["metadata"] == {"test": True}
        assert "/test" in info["endpoints"]
        assert info["endpoints"]["/test"]["method"] == "GET"
        assert info["endpoints"]["/test"]["description"] == "Test endpoint"
