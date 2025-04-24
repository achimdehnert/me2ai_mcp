"""
Tests for ME2AI MCP service discovery components.

This module contains tests for the service discovery
functionality of ME2AI MCP's microservice architecture.
"""

import pytest
import unittest.mock as mock
import asyncio
import time
import socket
from typing import Dict, Any, Optional, List

from me2ai_mcp.services.base import ServiceStatus, ServiceInfo
from me2ai_mcp.services.discovery import ServiceDiscovery


class TestServiceDiscovery:
    """Tests for the ServiceDiscovery class."""
    
    @mock.patch("me2ai_mcp.services.discovery.get_registry")
    def test_should_initialize_service_discovery(self, mock_get_registry):
        """Test that service discovery can be initialized."""
        # Mock registry
        registry = mock.MagicMock()
        mock_get_registry.return_value = registry
        
        # Create service discovery
        discovery = ServiceDiscovery(refresh_interval=30)
        
        # Verify properties
        assert discovery.refresh_interval == 30
        assert discovery.registry == registry
        assert discovery._refresh_task is None
        
    @mock.patch("me2ai_mcp.services.discovery.get_registry")
    @pytest.mark.asyncio
    async def test_should_start_service_discovery(self, mock_get_registry):
        """Test that service discovery can be started."""
        # Mock registry
        registry = mock.MagicMock()
        mock_get_registry.return_value = registry
        
        # Create service discovery
        discovery = ServiceDiscovery()
        
        # Start service discovery
        with mock.patch.object(asyncio, "create_task") as mock_create_task:
            result = await discovery.start()
        
        # Verify result
        assert result is True
        mock_create_task.assert_called_once()
        
    @mock.patch("me2ai_mcp.services.discovery.get_registry")
    @pytest.mark.asyncio
    async def test_should_stop_service_discovery(self, mock_get_registry):
        """Test that service discovery can be stopped."""
        # Mock registry
        registry = mock.MagicMock()
        mock_get_registry.return_value = registry
        
        # Create service discovery
        discovery = ServiceDiscovery()
        
        # Mock refresh task
        discovery._refresh_task = mock.MagicMock()
        discovery._refresh_task.cancel = mock.MagicMock()
        
        # Stop service discovery
        result = await discovery.stop()
        
        # Verify result
        assert result is True
        discovery._refresh_task.cancel.assert_called_once()
        
    @mock.patch("me2ai_mcp.services.discovery.get_registry")
    @pytest.mark.asyncio
    async def test_should_refresh_services(self, mock_get_registry):
        """Test that service discovery refreshes services."""
        # Mock registry
        registry = mock.MagicMock()
        mock_get_registry.return_value = registry
        
        # Create mock services
        service1 = ServiceInfo(
            id="test-service-1",
            name="TestService1",
            host="localhost",
            port=8000,
            status=ServiceStatus.RUNNING,
            version="0.1.0",
            endpoints={"/health": "GET"},
            metadata={"test": True}
        )
        
        service2 = ServiceInfo(
            id="test-service-2",
            name="TestService2",
            host="localhost",
            port=8001,
            status=ServiceStatus.RUNNING,
            version="0.1.0",
            endpoints={"/health": "GET"},
            metadata={"test": True}
        )
        
        # Mock registry.list_services to return our mock services
        registry.list_services.return_value = [service1, service2]
        
        # Create service discovery with short refresh interval
        discovery = ServiceDiscovery(refresh_interval=0.1)
        
        # Mock _check_service_health
        discovery._check_service_health = mock.AsyncMock(return_value=True)
        
        # Start refresh task
        refresh_task = asyncio.create_task(discovery._refresh_services())
        
        # Wait a moment for refresh to happen
        await asyncio.sleep(0.2)
        
        # Cancel task
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass
        
        # Verify services were checked
        assert discovery._check_service_health.await_count >= 2
        
    @mock.patch("me2ai_mcp.services.discovery.get_registry")
    def test_should_find_service(self, mock_get_registry):
        """Test that services can be found by name."""
        # Mock registry
        registry = mock.MagicMock()
        mock_get_registry.return_value = registry
        
        # Create mock service
        service = ServiceInfo(
            id="test-service-1",
            name="UniqueServiceName",
            host="localhost",
            port=8000,
            status=ServiceStatus.RUNNING,
            version="0.1.0",
            endpoints={"/health": "GET"},
            metadata={"test": True}
        )
        
        # Mock registry.get_service to return our mock service
        registry.get_service.return_value = service
        
        # Create service discovery
        discovery = ServiceDiscovery()
        
        # Find service
        result = discovery.find_service("UniqueServiceName")
        
        # Verify result
        assert result == service
        registry.get_service.assert_called_once_with("UniqueServiceName")
        
    @mock.patch("me2ai_mcp.services.discovery.get_registry")
    def test_should_find_services_by_endpoint(self, mock_get_registry):
        """Test that services can be found by endpoint."""
        # Mock registry
        registry = mock.MagicMock()
        mock_get_registry.return_value = registry
        
        # Create mock services
        service1 = ServiceInfo(
            id="test-service-1",
            name="TestService1",
            host="localhost",
            port=8000,
            status=ServiceStatus.RUNNING,
            version="0.1.0",
            endpoints={"/health": "GET"},
            metadata={"test": True}
        )
        
        service2 = ServiceInfo(
            id="test-service-2",
            name="TestService2",
            host="localhost",
            port=8001,
            status=ServiceStatus.RUNNING,
            version="0.1.0",
            endpoints={"/test": "POST", "/health": "GET"},
            metadata={"test": True}
        )
        
        # Mock registry.list_services to return our mock services
        registry.list_services.return_value = [service1, service2]
        
        # Create service discovery
        discovery = ServiceDiscovery()
        
        # Find services by endpoint
        results = discovery.find_services_by_endpoint("/health", "GET")
        
        # Verify result
        assert len(results) == 2
        assert service1 in results
        assert service2 in results
        
        # Find services by unique endpoint
        results = discovery.find_services_by_endpoint("/test", "POST")
        
        # Verify result
        assert len(results) == 1
        assert service2 in results
        
    @mock.patch("me2ai_mcp.services.discovery.get_registry")
    @mock.patch("me2ai_mcp.services.discovery.socket.socket")
    @pytest.mark.asyncio
    async def test_should_check_service_health(self, mock_socket, mock_get_registry):
        """Test that service health can be checked."""
        # Mock registry
        registry = mock.MagicMock()
        mock_get_registry.return_value = registry
        
        # Mock socket
        socket_instance = mock.MagicMock()
        mock_socket.return_value.__enter__.return_value = socket_instance
        socket_instance.connect_ex.return_value = 0  # Success
        
        # Create service discovery
        discovery = ServiceDiscovery()
        
        # Create mock service with recent heartbeat
        service = ServiceInfo(
            id="test-service-1",
            name="TestService",
            host="localhost",
            port=8000,
            status=ServiceStatus.RUNNING,
            version="0.1.0",
            endpoints={"/health": "GET"},
            metadata={"test": True},
            last_heartbeat=time.time()  # Recent heartbeat
        )
        
        # Check service health
        result = await discovery._check_service_health(service)
        
        # Verify result
        assert result is True
        socket_instance.connect_ex.assert_called_once_with(("localhost", 8000))
        
    @mock.patch("me2ai_mcp.services.discovery.get_registry")
    @mock.patch("me2ai_mcp.services.discovery.socket.socket")
    @pytest.mark.asyncio
    async def test_should_detect_unhealthy_service(self, mock_socket, mock_get_registry):
        """Test that unhealthy services are detected."""
        # Mock registry
        registry = mock.MagicMock()
        mock_get_registry.return_value = registry
        
        # Mock socket to fail connection
        socket_instance = mock.MagicMock()
        mock_socket.return_value.__enter__.return_value = socket_instance
        socket_instance.connect_ex.return_value = 1  # Failure
        
        # Create service discovery
        discovery = ServiceDiscovery()
        
        # Create mock service with recent heartbeat
        service = ServiceInfo(
            id="test-service-1",
            name="TestService",
            host="localhost",
            port=8000,
            status=ServiceStatus.RUNNING,
            version="0.1.0",
            endpoints={"/health": "GET"},
            metadata={"test": True},
            last_heartbeat=time.time()  # Recent heartbeat
        )
        
        # Check service health
        result = await discovery._check_service_health(service)
        
        # Verify result
        assert result is False
        socket_instance.connect_ex.assert_called_once_with(("localhost", 8000))
        
    @mock.patch("me2ai_mcp.services.discovery.get_registry")
    @pytest.mark.asyncio
    async def test_should_detect_stale_heartbeat(self, mock_get_registry):
        """Test that stale heartbeats are detected."""
        # Mock registry
        registry = mock.MagicMock()
        mock_get_registry.return_value = registry
        
        # Create service discovery
        discovery = ServiceDiscovery()
        
        # Create mock service with old heartbeat (4 minutes ago)
        service = ServiceInfo(
            id="test-service-1",
            name="TestService",
            host="localhost",
            port=8000,
            status=ServiceStatus.RUNNING,
            version="0.1.0",
            endpoints={"/health": "GET"},
            metadata={"test": True},
            last_heartbeat=time.time() - 240  # 4 minutes ago
        )
        
        # Check service health
        result = await discovery._check_service_health(service)
        
        # Verify result
        assert result is False
        
    @mock.patch("me2ai_mcp.services.discovery.get_registry")
    def test_should_list_services(self, mock_get_registry):
        """Test that all services can be listed."""
        # Mock registry
        registry = mock.MagicMock()
        mock_get_registry.return_value = registry
        
        # Create mock services
        services = [
            ServiceInfo(
                id="test-service-1",
                name="TestService1",
                host="localhost",
                port=8000,
                status=ServiceStatus.RUNNING,
                version="0.1.0",
                endpoints={"/health": "GET"},
                metadata={"test": True}
            ),
            ServiceInfo(
                id="test-service-2",
                name="TestService2",
                host="localhost",
                port=8001,
                status=ServiceStatus.RUNNING,
                version="0.1.0",
                endpoints={"/health": "GET"},
                metadata={"test": True}
            )
        ]
        
        # Mock registry.list_services to return our mock services
        registry.list_services.return_value = services
        
        # Create service discovery
        discovery = ServiceDiscovery()
        
        # List services
        result = discovery.list_services()
        
        # Verify result
        assert result == services
        registry.list_services.assert_called_once()
        
    @mock.patch("me2ai_mcp.services.discovery.get_registry")
    def test_should_get_service_status(self, mock_get_registry):
        """Test that service status can be retrieved."""
        # Mock registry
        registry = mock.MagicMock()
        mock_get_registry.return_value = registry
        
        # Create mock service
        service = ServiceInfo(
            id="test-service-1",
            name="TestService",
            host="localhost",
            port=8000,
            status=ServiceStatus.RUNNING,
            version="0.1.0",
            endpoints={"/health": "GET"},
            metadata={"test": True}
        )
        
        # Mock registry.get_service_by_id to return our mock service
        registry.get_service_by_id.return_value = service
        
        # Create service discovery
        discovery = ServiceDiscovery()
        
        # Get service status
        result = discovery.get_service_status("test-service-1")
        
        # Verify result
        assert result == ServiceStatus.RUNNING
        registry.get_service_by_id.assert_called_once_with("test-service-1")
