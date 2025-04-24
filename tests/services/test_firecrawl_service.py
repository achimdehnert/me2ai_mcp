"""
Tests for ME2AI MCP FireCrawl service.

This module contains tests for the FireCrawl microservice
implementation for ME2AI MCP.
"""

import pytest
import unittest.mock as mock
import asyncio
import os
import sys
import tempfile
from typing import Dict, Any, Optional, List, Tuple
from fastapi import FastAPI, Request, HTTPException

from me2ai_mcp.services.base import ServiceStatus
from me2ai_mcp.services.firecrawl_service import FireCrawlService, FIRECRAWL_REPO


# Mock FastAPI for testing
with mock.patch.dict("sys.modules", {
    "fastapi": mock.MagicMock(),
    "fastapi.FastAPI": mock.MagicMock(),
    "fastapi.APIRouter": mock.MagicMock(),
    "fastapi.Depends": mock.MagicMock(),
    "fastapi.HTTPException": mock.MagicMock(),
    "fastapi.Request": mock.MagicMock(),
    "fastapi.Response": mock.MagicMock(),
    "fastapi.Body": mock.MagicMock(),
    "fastapi.middleware.cors": mock.MagicMock(),
    "fastapi.responses": mock.MagicMock(),
    "uvicorn": mock.MagicMock()
}):
    # Force FASTAPI_AVAILABLE to be True for testing
    import me2ai_mcp.services.web
    me2ai_mcp.services.web.FASTAPI_AVAILABLE = True
    
    # Import firecrawl service with mocked modules
    from me2ai_mcp.services.firecrawl_service import FireCrawlService


@pytest.mark.asyncio
class TestFireCrawlService:
    """Tests for the FireCrawlService class."""
    
    async def test_should_initialize_firecrawl_service(self):
        """Test that the FireCrawl service can be initialized."""
        # Create service
        service = FireCrawlService(
            host="localhost",
            port=8787,
            version="0.1.0",
            firecrawl_path="/mock/firecrawl",
            auto_setup=False,
            javascript_enabled=True,
            default_wait_time=5,
            metadata={"test": True}
        )
        
        # Verify service properties
        assert service.name == "firecrawl"
        assert service.host == "localhost"
        assert service.port == 8787
        assert service.version == "0.1.0"
        assert service.firecrawl_path == "/mock/firecrawl"
        assert service.auto_setup is False
        assert service.javascript_enabled is True
        assert service.default_wait_time == 5
        assert "test" in service.metadata
        assert service.metadata["test"] is True
        assert service.browser_instances == {}
        
    @mock.patch("me2ai_mcp.services.firecrawl_service.os.path.exists")
    def test_should_find_firecrawl_path(self, mock_exists):
        """Test that FireCrawl path can be found if it exists."""
        # Mock os.path.exists to return True for a specific path
        def mock_exists_func(path):
            return "/mock/firecrawl" in path
            
        mock_exists.side_effect = mock_exists_func
        
        # Create service
        service = FireCrawlService(auto_setup=False)
        
        # Find FireCrawl path
        path = service._find_firecrawl_path()
        
        # Verify result
        assert path is not None
        assert "/mock/firecrawl" in path
        
    @mock.patch("me2ai_mcp.services.firecrawl_service.os.path.exists")
    def test_should_return_none_when_firecrawl_not_found(self, mock_exists):
        """Test that None is returned when FireCrawl path is not found."""
        # Mock os.path.exists to return False for all paths
        mock_exists.return_value = False
        
        # Create service
        service = FireCrawlService(auto_setup=False)
        
        # Find FireCrawl path
        path = service._find_firecrawl_path()
        
        # Verify result
        assert path is None
        
    @mock.patch("me2ai_mcp.services.firecrawl_service.asyncio.create_subprocess_exec")
    @mock.patch("me2ai_mcp.services.firecrawl_service.os.makedirs")
    @mock.patch("me2ai_mcp.services.firecrawl_service.tempfile.gettempdir")
    @pytest.mark.asyncio
    async def test_should_clone_firecrawl(
        self, mock_tempdir, mock_makedirs, mock_create_subprocess_exec
    ):
        """Test that FireCrawl can be cloned from GitHub."""
        # Mock tempfile.gettempdir
        mock_tempdir.return_value = "/mock/temp"
        
        # Mock subprocess execution
        process_mock = mock.MagicMock()
        process_mock.returncode = 0
        process_mock.communicate = mock.AsyncMock(
            return_value=(b"Cloning into '/mock/temp/firecrawl'...", b"")
        )
        mock_create_subprocess_exec.return_value = process_mock
        
        # Create service
        service = FireCrawlService(auto_setup=False)
        
        # Clone FireCrawl
        path = await service._clone_firecrawl()
        
        # Verify result
        assert path == "/mock/temp/firecrawl"
        mock_makedirs.assert_called_once_with("/mock/temp/firecrawl", exist_ok=True)
        mock_create_subprocess_exec.assert_called_once_with(
            "git", "clone", FIRECRAWL_REPO, "/mock/temp/firecrawl",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
    @mock.patch("me2ai_mcp.services.firecrawl_service.asyncio.create_subprocess_exec")
    @mock.patch("me2ai_mcp.services.firecrawl_service.os.path.exists")
    @pytest.mark.asyncio
    async def test_should_install_dependencies(self, mock_exists, mock_create_subprocess_exec):
        """Test that dependencies can be installed."""
        # Mock os.path.exists to return True for requirements.txt
        mock_exists.return_value = True
        
        # Mock subprocess execution
        process_mock = mock.MagicMock()
        process_mock.returncode = 0
        process_mock.communicate = mock.AsyncMock(
            return_value=(b"Successfully installed some-package", b"")
        )
        mock_create_subprocess_exec.return_value = process_mock
        
        # Create service
        service = FireCrawlService(
            auto_setup=False,
            firecrawl_path="/mock/firecrawl"
        )
        
        # Install dependencies
        result = await service._install_dependencies()
        
        # Verify result
        assert result is True
        mock_exists.assert_called_once_with("/mock/firecrawl/requirements.txt")
        mock_create_subprocess_exec.assert_called_once_with(
            sys.executable, "-m", "pip", "install", "-r", "/mock/firecrawl/requirements.txt",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
    @mock.patch("me2ai_mcp.services.firecrawl_service.os.path.exists")
    @mock.patch("me2ai_mcp.services.base.get_registry")
    @pytest.mark.asyncio
    async def test_should_setup_firecrawl(self, mock_get_registry, mock_exists):
        """Test that FireCrawl can be set up."""
        # Mock registry
        registry = mock.MagicMock()
        registry.register.return_value = True
        mock_get_registry.return_value = registry
        
        # Mock os.path.exists
        mock_exists.return_value = True
        
        # Create service
        service = FireCrawlService(auto_setup=False)
        
        # Mock required methods
        service._find_firecrawl_path = mock.MagicMock(return_value="/mock/firecrawl")
        service._install_dependencies = mock.AsyncMock(return_value=True)
        
        # Set up FireCrawl
        result = await service._setup_firecrawl()
        
        # Verify result
        assert result is True
        assert service.firecrawl_path == "/mock/firecrawl"
        service._find_firecrawl_path.assert_called_once()
        service._install_dependencies.assert_awaited_once()
        
    @mock.patch("me2ai_mcp.services.firecrawl_service.os.path.exists")
    @mock.patch("me2ai_mcp.services.base.get_registry")
    @pytest.mark.asyncio
    async def test_should_start_service(self, mock_get_registry, mock_exists):
        """Test that the FireCrawl service can be started."""
        # Mock registry
        registry = mock.MagicMock()
        registry.register.return_value = True
        mock_get_registry.return_value = registry
        
        # Mock os.path.exists
        mock_exists.return_value = True
        
        # Create service
        service = FireCrawlService(auto_setup=True)
        
        # Mock required methods
        service._setup_firecrawl = mock.AsyncMock(return_value=True)
        service._send_heartbeat = mock.AsyncMock()
        
        # Mock server
        service._server = mock.MagicMock()
        service._server.serve = mock.AsyncMock()
        
        # Start service
        with mock.patch.object(asyncio, "create_task") as mock_create_task:
            result = await service.start()
        
        # Verify result
        assert result is True
        assert service.status == ServiceStatus.RUNNING
        service._setup_firecrawl.assert_awaited_once()
        mock_create_task.assert_called()
        
    @mock.patch("me2ai_mcp.services.base.get_registry")
    @pytest.mark.asyncio
    async def test_should_stop_service(self, mock_get_registry):
        """Test that the FireCrawl service can be stopped."""
        # Mock registry
        registry = mock.MagicMock()
        registry.unregister.return_value = True
        mock_get_registry.return_value = registry
        
        # Create service
        service = FireCrawlService(auto_setup=False)
        
        # Mock browser instances
        process1 = mock.MagicMock()
        process1.terminate = mock.MagicMock()
        
        process2 = mock.MagicMock()
        process2.terminate = mock.MagicMock()
        
        service.browser_instances = {
            "browser1": {"process": process1},
            "browser2": {"process": process2}
        }
        
        # Set service state
        service.status = ServiceStatus.RUNNING
        service.start_time = 0
        
        # Mock server
        service._server = mock.MagicMock()
        service._server.should_exit = False
        service._server_task = mock.MagicMock()
        service._server_task.__await__ = mock.MagicMock(return_value=iter([None]))
        
        # Mock heartbeat task
        service._heartbeat_task = mock.MagicMock()
        service._heartbeat_task.cancel = mock.MagicMock()
        
        # Stop service
        result = await service.stop()
        
        # Verify result
        assert result is True
        assert service.status == ServiceStatus.STOPPED
        
        # Verify browser instances were terminated
        process1.terminate.assert_called_once()
        process2.terminate.assert_called_once()
        
        # Verify server was stopped
        assert service._server.should_exit is True
        
        # Verify registry interactions
        registry.unregister.assert_called_once_with(service.id)
        
    @pytest.mark.asyncio
    async def test_should_handle_scrape_request(self):
        """Test that scrape requests can be handled."""
        # Create service
        service = FireCrawlService(auto_setup=False)
        
        # Create mock request
        request = mock.MagicMock()
        
        # Create parameters
        params = {
            "url": "https://example.com",
            "javascript_enabled": True,
            "wait_time": 5,
            "timeout": 60,
            "headers": {"User-Agent": "Test Agent"}
        }
        
        # Handle scrape request
        result = await service.handle_scrape(request, params)
        
        # Verify result format
        assert "scrape_id" in result
        assert result["url"] == "https://example.com"
        assert "title" in result
        assert "content" in result
        assert "links" in result
        assert "metadata" in result
        assert result["metadata"]["javascript_enabled"] is True
        assert result["metadata"]["wait_time"] == 5
        assert result["metadata"]["timeout"] == 60
        assert result["metadata"]["headers"] == {"User-Agent": "Test Agent"}
        
    @pytest.mark.asyncio
    async def test_should_handle_crawl_request(self):
        """Test that crawl requests can be handled."""
        # Create service
        service = FireCrawlService(auto_setup=False)
        
        # Create mock request
        request = mock.MagicMock()
        
        # Create parameters
        params = {
            "url": "https://example.com",
            "max_depth": 3,
            "max_pages": 30,
            "javascript_enabled": True,
            "wait_time": 5
        }
        
        # Handle crawl request
        result = await service.handle_crawl(request, params)
        
        # Verify result format
        assert "crawl_id" in result
        assert result["url"] == "https://example.com"
        assert result["max_depth"] == 3
        assert result["max_pages"] == 30
        assert "pages_crawled" in result
        assert "results" in result
        assert len(result["results"]) > 0
        assert "metadata" in result
        assert result["metadata"]["javascript_enabled"] is True
        assert result["metadata"]["wait_time"] == 5
        
    @pytest.mark.asyncio
    async def test_should_handle_status_request(self):
        """Test that status requests can be handled."""
        # Create service
        service = FireCrawlService(
            auto_setup=False,
            firecrawl_path="/mock/firecrawl"
        )
        
        # Set service state
        service.status = ServiceStatus.RUNNING
        service.start_time = 0
        
        # Create mock request
        request = mock.MagicMock()
        
        # Handle status request
        result = await service.handle_status(request)
        
        # Verify result format
        assert "status" in result
        assert result["status"] == "running"
        assert "uptime" in result
        assert "version" in result
        assert "firecrawl_path" in result
        assert result["firecrawl_path"] == "/mock/firecrawl"
        assert "active_browsers" in result
        assert "javascript_enabled" in result
        assert "default_wait_time" in result
        
    @pytest.mark.asyncio
    async def test_should_handle_screenshot_request(self):
        """Test that screenshot requests can be handled."""
        # Create service
        service = FireCrawlService(auto_setup=False)
        
        # Create mock request
        request = mock.MagicMock()
        
        # Create parameters
        params = {
            "url": "https://example.com",
            "wait_time": 5,
            "full_page": True
        }
        
        # Handle screenshot request
        result = await service.handle_screenshot(request, params)
        
        # Verify result format
        assert "screenshot_id" in result
        assert result["url"] == "https://example.com"
        assert "timestamp" in result
        assert "image_format" in result
        assert "image_data" in result
        assert "metadata" in result
        assert result["metadata"]["wait_time"] == 5
        assert result["metadata"]["full_page"] is True
        
    @pytest.mark.asyncio
    async def test_should_reject_invalid_url(self):
        """Test that invalid URLs are rejected."""
        # Create service
        service = FireCrawlService(auto_setup=False)
        
        # Create mock request
        request = mock.MagicMock()
        
        # Create parameters with invalid URL
        params = {
            "url": "invalid-url"
        }
        
        # Handle scrape request should raise HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await service.handle_scrape(request, params)
            
        # Verify exception
        assert excinfo.value.status_code == 400
        assert "Invalid URL scheme" in str(excinfo.value.detail)
        
    @pytest.mark.asyncio
    async def test_should_reject_missing_url(self):
        """Test that missing URLs are rejected."""
        # Create service
        service = FireCrawlService(auto_setup=False)
        
        # Create mock request
        request = mock.MagicMock()
        
        # Create parameters without URL
        params = {
            "javascript_enabled": True
        }
        
        # Handle scrape request should raise HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await service.handle_scrape(request, params)
            
        # Verify exception
        assert excinfo.value.status_code == 400
        assert "URL parameter is required" in str(excinfo.value.detail)
        
    @pytest.mark.asyncio
    async def test_should_provide_enhanced_health_check(self):
        """Test that health check includes FireCrawl-specific information."""
        # Create service
        service = FireCrawlService(
            auto_setup=False,
            firecrawl_path="/mock/firecrawl"
        )
        
        # Set service state
        service.status = ServiceStatus.RUNNING
        service.start_time = 0
        
        # Get health check
        health = await service.health_check()
        
        # Verify health check format
        assert health["status"] == "running"
        assert "uptime" in health
        assert "version" in health
        assert "firecrawl_available" in health
        assert health["firecrawl_available"] is True
