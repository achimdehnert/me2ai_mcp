"""
Web service base for ME2AI MCP.

This module provides the foundation for HTTP-based microservices
within the ME2AI MCP framework, including request handling, routing,
and standard middleware.
"""

from typing import Dict, List, Any, Optional, Union, Type, Callable, Awaitable, Set
import logging
import asyncio
import json
import time
import uuid
import os
from dataclasses import dataclass, field
import inspect
import traceback
from functools import wraps

# FastAPI components for web services
try:
    import fastapi
    from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = object  # type: ignore
    APIRouter = object  # type: ignore
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. Web services will not function.")

# Import base service components
from .base import BaseService, ServiceStatus, ServiceEndpoint

# Configure logging
logger = logging.getLogger("me2ai-mcp-web-services")


def requires_fastapi(func):
    """Decorator to check if FastAPI is available before executing a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for web services")
        return func(*args, **kwargs)
    return wrapper


class WebService(BaseService):
    """
    Base class for HTTP-based microservices.
    
    This class extends BaseService to provide HTTP request handling
    capabilities using FastAPI.
    """
    
    def __init__(
        self,
        name: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        version: str = "0.1.0",
        metadata: Optional[Dict[str, Any]] = None,
        enable_cors: bool = True,
        cors_origins: List[str] = None,
        enable_docs: bool = True,
        docs_url: str = "/docs",
        openapi_url: str = "/openapi.json",
        enable_middleware: bool = True
    ):
        """
        Initialize the web service.
        
        Args:
            name: Service name
            host: Host to bind the service to
            port: Port to bind the service to
            version: Service version
            metadata: Additional service metadata
            enable_cors: Whether to enable CORS
            cors_origins: List of allowed CORS origins (defaults to ["*"])
            enable_docs: Whether to enable API documentation
            docs_url: URL for API documentation
            openapi_url: URL for OpenAPI schema
            enable_middleware: Whether to enable standard middleware
        """
        super().__init__(name, host, port, version, metadata)
        
        # Initialize FastAPI
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title=f"{name} Service",
                description=f"ME2AI MCP {name} Service",
                version=version,
                docs_url=docs_url if enable_docs else None,
                openapi_url=openapi_url if enable_docs else None
            )
            
            # Set up CORS
            if enable_cors:
                self.app.add_middleware(
                    CORSMiddleware,
                    allow_origins=cors_origins or ["*"],
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"]
                )
                
            # Add standard middleware
            if enable_middleware:
                self._setup_middleware()
                
            # Register built-in routes
            self._register_builtin_routes()
        else:
            self.app = None
            
        # Initialize server instance
        self._server = None
    
    @requires_fastapi
    def _setup_middleware(self) -> None:
        """Set up standard middleware for the service."""
        @self.app.middleware("http")
        async def logging_middleware(request: Request, call_next):
            """Log request and response details."""
            start_time = time.time()
            request_id = str(uuid.uuid4())
            
            # Log request
            self.logger.info(
                f"Request {request_id}: {request.method} {request.url.path}"
            )
            
            try:
                # Process request
                response = await call_next(request)
                
                # Log response
                process_time = (time.time() - start_time) * 1000
                self.logger.info(
                    f"Response {request_id}: {response.status_code} "
                    f"({process_time:.2f}ms)"
                )
                
                # Add headers
                response.headers["X-Process-Time"] = f"{process_time:.2f}"
                response.headers["X-Request-ID"] = request_id
                
                return response
                
            except Exception as e:
                # Log error
                self.logger.error(
                    f"Error {request_id}: {str(e)}\n{traceback.format_exc()}"
                )
                
                # Return error response
                process_time = (time.time() - start_time) * 1000
                error_response = JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "error": str(e),
                        "request_id": request_id
                    }
                )
                error_response.headers["X-Process-Time"] = f"{process_time:.2f}"
                error_response.headers["X-Request-ID"] = request_id
                
                return error_response
    
    @requires_fastapi
    def _register_builtin_routes(self) -> None:
        """Register built-in routes for the service."""
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return await self.health_check()
            
        @self.app.get("/info")
        async def get_info():
            """Service information endpoint."""
            return await self.get_info()
    
    @requires_fastapi
    def register_route(
        self,
        path: str,
        method: str,
        handler: Callable,
        description: str,
        response_model: Optional[Type] = None,
        auth_required: bool = False,
        rate_limit: Optional[int] = None
    ) -> None:
        """
        Register a route with the service.
        
        Args:
            path: URL path for the route
            method: HTTP method (GET, POST, etc.)
            handler: Function to handle the route
            description: Description of the route
            response_model: Pydantic model for the response
            auth_required: Whether authentication is required
            rate_limit: Rate limit for the route in requests per minute
        """
        # Register with endpoint registry
        parameters = {}
        
        # Extract parameters from handler signature
        signature = inspect.signature(handler)
        for param_name, param in signature.parameters.items():
            if param_name != "self" and param.annotation is not inspect.Parameter.empty:
                parameters[param_name] = {
                    "type": str(param.annotation),
                    "default": None if param.default is inspect.Parameter.empty else param.default
                }
                
        # Register endpoint
        self.register_endpoint(
            path=path,
            method=method,
            description=description,
            parameters=parameters,
            response_schema=None,  # Could extract from response_model
            auth_required=auth_required,
            rate_limit=rate_limit
        )
        
        # Add route to FastAPI
        method = method.lower()
        route_handler = getattr(self.app, method, None)
        
        if route_handler is None:
            self.logger.error(f"Invalid HTTP method: {method}")
            return
            
        if response_model:
            route_handler(path, response_model=response_model)(handler)
        else:
            route_handler(path)(handler)
            
        self.logger.debug(f"Registered route: {method.upper()} {path}")
    
    async def start(self) -> bool:
        """
        Start the web service.
        
        Returns:
            bool: True if the service started successfully
        """
        if not FASTAPI_AVAILABLE:
            self.logger.error("FastAPI is required for web services")
            self.status = ServiceStatus.ERROR
            return False
            
        try:
            # Call parent start method
            result = await super().start()
            if not result:
                return False
                
            # Start FastAPI server
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
            
            self._server = uvicorn.Server(config)
            
            # Run server in background task
            self._server_task = asyncio.create_task(self._server.serve())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting web service: {str(e)}")
            self.status = ServiceStatus.ERROR
            return False
    
    async def stop(self) -> bool:
        """
        Stop the web service.
        
        Returns:
            bool: True if the service stopped successfully
        """
        try:
            # Stop FastAPI server
            if self._server is not None:
                self._server.should_exit = True
                
                if hasattr(self, "_server_task"):
                    await self._server_task
                    
            # Call parent stop method
            return await super().stop()
            
        except Exception as e:
            self.logger.error(f"Error stopping web service: {str(e)}")
            self.status = ServiceStatus.ERROR
            return False
