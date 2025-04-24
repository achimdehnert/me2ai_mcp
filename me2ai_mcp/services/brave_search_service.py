"""
Brave Search service for ME2AI MCP.

This module provides a microservice implementation of the Brave Search
service for ME2AI MCP, offering enhanced web search capabilities.
"""

from typing import Dict, List, Any, Optional, Union, Literal
import logging
import asyncio
import json
import os
import time
import uuid
import re
from dataclasses import dataclass, field
from enum import Enum

# Import service components
from .web import WebService
from me2ai_mcp.services.base import ServiceStatus

try:
    import fastapi
    from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request, Response, Body, Query
    from fastapi.responses import JSONResponse
    import uvicorn
    import pydantic
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. Brave Search service will not function.")
    BaseModel = object  # type: ignore

# Configure logging
logger = logging.getLogger("me2ai-mcp-brave-search-service")

# Default Brave Search API endpoint
BRAVE_SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_PORT = 8788


class SearchType(str, Enum):
    """Types of searches supported by Brave Search."""
    
    WEB = "web"
    NEWS = "news"
    VIDEOS = "videos"
    IMAGES = "images"


class SearchRequest(BaseModel):
    """Request model for search requests."""
    
    query: str = Field(..., description="Search query")
    country: Optional[str] = Field(None, description="Country code (e.g., 'us', 'de')")
    search_type: SearchType = Field(default=SearchType.WEB, description="Type of search")
    count: Optional[int] = Field(None, description="Number of results (max 20)")
    offset: Optional[int] = Field(None, description="Offset for pagination")
    freshness: Optional[str] = Field(None, description="Time range (e.g., 'pd' for past day)")
    safe_search: Optional[bool] = Field(None, description="Whether to enable SafeSearch")


class BraveSearchService(WebService):
    """
    Microservice for web search using Brave Search API.
    
    This service provides enhanced web search capabilities through
    the Brave Search API, offering privacy-focused and comprehensive
    search results.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        version: str = "0.1.0",
        api_key: Optional[str] = None,
        default_country: str = "us",
        default_count: int = 10,
        safe_search: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Brave Search service.
        
        Args:
            host: Host to bind the service to
            port: Port to bind the service to
            version: Service version
            api_key: Brave Search API key
            default_country: Default country for search results
            default_count: Default number of results per request
            safe_search: Whether to enable SafeSearch by default
            metadata: Additional service metadata
        """
        # Set up metadata
        metadata = metadata or {}
        metadata.update({
            "default_country": default_country,
            "default_count": default_count,
            "safe_search": safe_search
        })
        
        # Initialize base web service
        super().__init__(
            name="brave_search", 
            host=host, 
            port=port, 
            version=version,
            metadata=metadata,
            enable_cors=True,
            cors_origins=["*"],
            enable_docs=True
        )
        
        # Set up Brave Search properties
        self.api_key = api_key or os.environ.get("BRAVE_SEARCH_API_KEY")
        self.default_country = default_country
        self.default_count = default_count
        self.safe_search = safe_search
        
        # Cache for rate limiting
        self.request_cache = {}
        
        # Register service endpoints
        self._register_service_endpoints()
    
    def _register_service_endpoints(self) -> None:
        """Register Brave Search service endpoints."""
        if not FASTAPI_AVAILABLE:
            self.logger.error("FastAPI is required for web services")
            return
            
        # Register search endpoint
        self.register_route(
            path="/search",
            method="POST",
            handler=self.handle_search,
            description="Search the web using Brave Search API"
        )
        
        # Register web search endpoint (GET)
        self.register_route(
            path="/web",
            method="GET",
            handler=self.handle_web_search,
            description="Search the web using Brave Search API (GET method)"
        )
        
        # Register news search endpoint
        self.register_route(
            path="/news",
            method="GET",
            handler=self.handle_news_search,
            description="Search for news using Brave Search API"
        )
        
        # Register video search endpoint
        self.register_route(
            path="/videos",
            method="GET",
            handler=self.handle_video_search,
            description="Search for videos using Brave Search API"
        )
        
        # Register image search endpoint
        self.register_route(
            path="/images",
            method="GET",
            handler=self.handle_image_search,
            description="Search for images using Brave Search API"
        )
    
    async def start(self) -> bool:
        """
        Start the Brave Search service.
        
        Returns:
            bool: True if the service started successfully
        """
        if not self.api_key:
            self.logger.warning("No Brave Search API key provided")
            self.metadata["api_key_status"] = "missing"
            
        # Start the base web service
        return await super().start()
    
    async def handle_search(
        self,
        request: Request,
        params: SearchRequest = Body(...)
    ) -> Dict[str, Any]:
        """
        Handle a search request.
        
        Args:
            request: FastAPI request object
            params: Search parameters
            
        Returns:
            Dict[str, Any]: Search results
        """
        if not self.api_key:
            raise HTTPException(
                status_code=500,
                detail="Brave Search API key not configured"
            )
            
        query = params.query
        country = params.country or self.default_country
        search_type = params.search_type
        count = params.count or self.default_count
        offset = params.offset or 0
        freshness = params.freshness
        safe_search = params.safe_search if params.safe_search is not None else self.safe_search
        
        try:
            # Call Brave Search API
            results = await self._call_brave_search(
                query=query,
                country=country,
                search_type=search_type,
                count=count,
                offset=offset,
                freshness=freshness,
                safe_search=safe_search
            )
            
            return results
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error handling search request: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error searching: {str(e)}"
            )
    
    async def handle_web_search(
        self,
        request: Request,
        q: str = Query(..., description="Search query"),
        country: Optional[str] = Query(None, description="Country code"),
        count: Optional[int] = Query(None, description="Number of results"),
        offset: Optional[int] = Query(None, description="Offset for pagination"),
        freshness: Optional[str] = Query(None, description="Time range"),
        safe: Optional[bool] = Query(None, description="Whether to enable SafeSearch")
    ) -> Dict[str, Any]:
        """
        Handle a web search request.
        
        Args:
            request: FastAPI request object
            q: Search query
            country: Country code
            count: Number of results
            offset: Offset for pagination
            freshness: Time range
            safe: Whether to enable SafeSearch
            
        Returns:
            Dict[str, Any]: Search results
        """
        params = SearchRequest(
            query=q,
            country=country,
            search_type=SearchType.WEB,
            count=count,
            offset=offset,
            freshness=freshness,
            safe_search=safe
        )
        
        return await self.handle_search(request, params)
    
    async def handle_news_search(
        self,
        request: Request,
        q: str = Query(..., description="Search query"),
        country: Optional[str] = Query(None, description="Country code"),
        count: Optional[int] = Query(None, description="Number of results"),
        offset: Optional[int] = Query(None, description="Offset for pagination"),
        freshness: Optional[str] = Query(None, description="Time range"),
        safe: Optional[bool] = Query(None, description="Whether to enable SafeSearch")
    ) -> Dict[str, Any]:
        """
        Handle a news search request.
        
        Args:
            request: FastAPI request object
            q: Search query
            country: Country code
            count: Number of results
            offset: Offset for pagination
            freshness: Time range
            safe: Whether to enable SafeSearch
            
        Returns:
            Dict[str, Any]: Search results
        """
        params = SearchRequest(
            query=q,
            country=country,
            search_type=SearchType.NEWS,
            count=count,
            offset=offset,
            freshness=freshness,
            safe_search=safe
        )
        
        return await self.handle_search(request, params)
    
    async def handle_video_search(
        self,
        request: Request,
        q: str = Query(..., description="Search query"),
        country: Optional[str] = Query(None, description="Country code"),
        count: Optional[int] = Query(None, description="Number of results"),
        offset: Optional[int] = Query(None, description="Offset for pagination"),
        freshness: Optional[str] = Query(None, description="Time range"),
        safe: Optional[bool] = Query(None, description="Whether to enable SafeSearch")
    ) -> Dict[str, Any]:
        """
        Handle a video search request.
        
        Args:
            request: FastAPI request object
            q: Search query
            country: Country code
            count: Number of results
            offset: Offset for pagination
            freshness: Time range
            safe: Whether to enable SafeSearch
            
        Returns:
            Dict[str, Any]: Search results
        """
        params = SearchRequest(
            query=q,
            country=country,
            search_type=SearchType.VIDEOS,
            count=count,
            offset=offset,
            freshness=freshness,
            safe_search=safe
        )
        
        return await self.handle_search(request, params)
    
    async def handle_image_search(
        self,
        request: Request,
        q: str = Query(..., description="Search query"),
        country: Optional[str] = Query(None, description="Country code"),
        count: Optional[int] = Query(None, description="Number of results"),
        offset: Optional[int] = Query(None, description="Offset for pagination"),
        freshness: Optional[str] = Query(None, description="Time range"),
        safe: Optional[bool] = Query(None, description="Whether to enable SafeSearch")
    ) -> Dict[str, Any]:
        """
        Handle an image search request.
        
        Args:
            request: FastAPI request object
            q: Search query
            country: Country code
            count: Number of results
            offset: Offset for pagination
            freshness: Time range
            safe: Whether to enable SafeSearch
            
        Returns:
            Dict[str, Any]: Search results
        """
        params = SearchRequest(
            query=q,
            country=country,
            search_type=SearchType.IMAGES,
            count=count,
            offset=offset,
            freshness=freshness,
            safe_search=safe
        )
        
        return await self.handle_search(request, params)
    
    async def _call_brave_search(
        self,
        query: str,
        country: str,
        search_type: SearchType,
        count: int = 10,
        offset: int = 0,
        freshness: Optional[str] = None,
        safe_search: bool = True
    ) -> Dict[str, Any]:
        """
        Call the Brave Search API.
        
        Args:
            query: Search query
            country: Country code
            search_type: Type of search
            count: Number of results
            offset: Offset for pagination
            freshness: Time range
            safe_search: Whether to enable SafeSearch
            
        Returns:
            Dict[str, Any]: Search results
            
        Raises:
            HTTPException: If the API call fails
        """
        # Import aiohttp here to avoid circular imports
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp is required for Brave Search")
            
        # Enforce rate limits
        await self._enforce_rate_limit(query)
        
        # Prepare API URL based on search type
        api_url = BRAVE_SEARCH_API_URL
        if search_type != SearchType.WEB:
            api_url = api_url.replace("/web/", f"/{search_type.value}/")
            
        # Prepare request parameters
        params = {
            "q": query,
            "country": country,
            "count": min(count, 20),  # API limits to 20 results per request
            "offset": offset
        }
        
        # Add optional parameters
        if freshness:
            params["freshness"] = freshness
            
        if safe_search is not None:
            params["safesearch"] = str(int(safe_search))
            
        # Prepare headers
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    api_url,
                    params=params,
                    headers=headers,
                    timeout=30
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Brave Search API error: {error_text}")
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Brave Search API error: {error_text}"
                        )
                        
                    # Parse response
                    data = await response.json()
                    
                    # Return search results with additional metadata
                    return {
                        "query": query,
                        "search_type": search_type,
                        "timestamp": time.time(),
                        "total_results": data.get("totalResults", 0),
                        "results": data.get("results", []),
                        "meta": {
                            "query_time_ms": data.get("queryTimeMs", 0),
                            "country": country,
                            "safe_search": safe_search,
                            "offset": offset,
                            "count": count,
                            "freshness": freshness
                        }
                    }
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"Brave Search API client error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error connecting to Brave Search API: {str(e)}"
            )
            
        except asyncio.TimeoutError:
            self.logger.error("Brave Search API timeout")
            raise HTTPException(
                status_code=504,
                detail="Brave Search API timeout"
            )
            
        except Exception as e:
            self.logger.error(f"Unexpected error calling Brave Search API: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )
    
    async def _enforce_rate_limit(self, query: str) -> None:
        """
        Enforce rate limiting for Brave Search API requests.
        
        Args:
            query: Search query
            
        Raises:
            HTTPException: If rate limit is exceeded
        """
        # Normalize query for caching
        normalized_query = re.sub(r'\s+', ' ', query.strip().lower())
        
        # Check if query has been recently searched
        current_time = time.time()
        if normalized_query in self.request_cache:
            last_request_time, count = self.request_cache[normalized_query]
            
            # If within 10 seconds, increment count
            if current_time - last_request_time < 10:
                if count >= 5:
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded for this query"
                    )
                self.request_cache[normalized_query] = (last_request_time, count + 1)
            else:
                # Reset if more than 10 seconds have passed
                self.request_cache[normalized_query] = (current_time, 1)
        else:
            # New query
            self.request_cache[normalized_query] = (current_time, 1)
            
        # Clean up old cache entries
        self._clean_request_cache()
    
    def _clean_request_cache(self) -> None:
        """Clean up old request cache entries."""
        current_time = time.time()
        to_remove = []
        
        for query, (timestamp, _) in self.request_cache.items():
            if current_time - timestamp > 60:  # Remove entries older than 60 seconds
                to_remove.append(query)
                
        for query in to_remove:
            del self.request_cache[query]
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check for the service.
        
        Returns:
            Dict[str, Any]: Health check information
        """
        # Get base health check
        health = await super().health_check()
        
        # Add Brave Search-specific checks
        health["api_key_configured"] = bool(self.api_key)
        health["cache_size"] = len(self.request_cache)
        
        return health
