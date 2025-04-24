"""
Microservices package for ME2AI MCP.

This package provides microservice implementations for various components
of the ME2AI MCP system, enabling scalable and extensible knowledge management
through a distributed architecture.
"""

# Base service architecture
from me2ai_mcp.services.base import (
    BaseService,
    ServiceRegistry,
    ServiceInfo,
    ServiceEndpoint,
    ServiceStatus,
    ServiceClient,
    ServiceManager
)

# Web service framework
from me2ai_mcp.services.web import WebService

# Service discovery
from me2ai_mcp.services.discovery import ServiceDiscovery

# Import available service implementations
from me2ai_mcp.services.firecrawl_service import FireCrawlService
from me2ai_mcp.services.brave_search_service import BraveSearchService

# Try to import vector store service (requires optional dependencies)
try:
    from me2ai_mcp.services.vectorstore_service import (
        VectorStoreService, 
        VectorStoreType,
        EmbeddingModel
    )
    VECTORSTORE_AVAILABLE = True
except ImportError:
    VECTORSTORE_AVAILABLE = False

__all__ = [
    # Core service architecture
    'BaseService',
    'ServiceRegistry',
    'ServiceInfo',
    'ServiceEndpoint',
    'ServiceStatus',
    'ServiceClient',
    'ServiceManager',
    'WebService',
    'ServiceDiscovery',
    
    # Service implementations
    'FireCrawlService',
    'BraveSearchService',
]

# Add optional services
if VECTORSTORE_AVAILABLE:
    __all__.extend([
        'VectorStoreService',
        'VectorStoreType',
        'EmbeddingModel'
    ])
