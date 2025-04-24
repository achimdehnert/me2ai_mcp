"""
Tests for ME2AI MCP web service components.

This module contains tests for the web service architecture
components of ME2AI MCP.
"""

import pytest
import unittest.mock as mock
import asyncio
import time
from typing import Dict, Any, Optional

from me2ai_mcp.services.base import ServiceStatus
from me2ai_mcp.services.web import WebService, requires_fastapi

# Mock FastAPI for testing
with mock.patch.dict("sys.modules", {
    "fastapi": mock.MagicMock(),
    "fastapi.FastAPI": mock.MagicMock(),
    "fastapi.APIRouter": mock.MagicMock(),
    "fastapi.Depends": mock.MagicMock(),
    "fastapi.HTTPException": mock.MagicMock(),
    "fastapi.Request": mock.MagicMock(),
    "fastapi.Response": mock.MagicMock(),
    "fastapi.middleware.cors": mock.MagicMock(),
    "fastapi.responses": mock.MagicMock(),
    "uvicorn": mock.MagicMock()
}):
    # Force FASTAPI_AVAILABLE to be True for testing
    import me2ai_mcp.services.web
    me2ai_mcp.services.web.FASTAPI_AVAILABLE = True
    
    # Import again with mocked modules
    from me2ai_mcp.services.web import WebService


@pytest.mark.asyncio
class TestWebService:
    """Tests for the WebService class."""
    
    async def test_should_initialize_web_service(self):
        """Test that web services can be initialized properly."""
        # Create service
        service = WebService(
            name="TestWebService",
            host="localhost",
            port=8000,
            version="0.1.0",
            metadata={"test": True},
            enable_cors=True,
            enable_docs=True
        )
        
        # Verify service properties
        assert service.name == "TestWebService"
        assert service.host == "localhost"
        assert service.port == 8000
        assert service.version == "0.1.0"
        assert service.metadata == {"test": True}
        assert service.status == ServiceStatus.INITIALIZING
        assert service.app is not None
        
    async def test_should_register_route(self):
        """Test that routes can be registered."""
        # Create service
        service = WebService(
            name="TestWebService",
            host="localhost",
            port=8000
        )
        
        # Mock the FastAPI app
        service.app.get = mock.MagicMock()
        
        # Define a handler function
        async def test_handler():
            return {"message": "Hello, world!"}
        
        # Register the route
        service.register_route(
            path="/test",
            method="GET",
            handler=test_handler,
            description="Test endpoint",
            auth_required=False
        )
        
        # Verify the endpoint was registered
        assert "GET/test" in service.endpoints
        endpoint = service.endpoints["GET/test"]
        assert endpoint.path == "/test"
        assert endpoint.method == "GET"
        assert endpoint.description == "Test endpoint"
        
        # Verify the route was added to FastAPI
        service.app.get.assert_called_once()
        
    @mock.patch("me2ai_mcp.services.base.get_registry")
    async def test_should_start_web_service(self, mock_get_registry):
        """Test that web services can be started."""
        # Mock service registry
        registry = mock.MagicMock()
        registry.register.return_value = True
        mock_get_registry.return_value = registry
        
        # Create service
        service = WebService(
            name="TestWebService",
            host="localhost",
            port=8000
        )
        
        # Mock uvicorn server
        service._server = mock.MagicMock()
        service._server.serve = mock.AsyncMock()
        
        # Start the service
        with mock.patch.object(service, "_send_heartbeat", return_value=None):
            service._send_heartbeat = mock.AsyncMock()
            with mock.patch.object(asyncio, "create_task") as mock_create_task:
                result = await service.start()
        
        # Verify the result
        assert result is True
        assert service.status == ServiceStatus.RUNNING
        assert service.start_time is not None
        
        # Verify registry interactions
        registry.register.assert_called_once()
        
        # Verify server was started
        mock_create_task.assert_called()
        
    @mock.patch("me2ai_mcp.services.base.get_registry")
    async def test_should_stop_web_service(self, mock_get_registry):
        """Test that web services can be stopped."""
        # Mock service registry
        registry = mock.MagicMock()
        registry.unregister.return_value = True
        mock_get_registry.return_value = registry
        
        # Create service
        service = WebService(
            name="TestWebService",
            host="localhost",
            port=8000
        )
        
        # Set service state
        service.status = ServiceStatus.RUNNING
        service.start_time = time.time()
        
        # Mock server
        service._server = mock.MagicMock()
        service._server_task = mock.MagicMock()
        service._server_task.__await__ = mock.MagicMock(return_value=iter([None]))
        
        # Add heartbeat task mock
        service._heartbeat_task = mock.MagicMock()
        service._heartbeat_task.cancel = mock.MagicMock()
        
        # Stop the service
        result = await service.stop()
        
        # Verify the result
        assert result is True
        assert service.status == ServiceStatus.STOPPED
        
        # Verify server was stopped
        assert service._server.should_exit is True
        
        # Verify registry interactions
        registry.unregister.assert_called_once_with(service.id)
        
    @mock.patch("me2ai_mcp.services.web.requires_fastapi")
    async def test_should_handle_fastapi_requirement(self, mock_requires_fastapi):
        """Test that the requires_fastapi decorator works correctly."""
        # Make the decorator pass through the function
        mock_requires_fastapi.side_effect = lambda f: f
        
        # Test function to decorate
        @requires_fastapi
        def test_function():
            return "test_result"
        
        # Verify without FastAPI
        with mock.patch("me2ai_mcp.services.web.FASTAPI_AVAILABLE", False):
            with pytest.raises(ImportError):
                test_function()
        
        # Verify with FastAPI
        with mock.patch("me2ai_mcp.services.web.FASTAPI_AVAILABLE", True):
            result = test_function()
            assert result == "test_result"
