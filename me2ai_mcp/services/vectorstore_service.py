"""
Vector Store service for ME2AI MCP.

This module provides a microservice implementation of the Vector Store
service for ME2AI MCP, offering document embedding and semantic search capabilities.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
import asyncio
import json
import os
import time
import uuid
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# Import backend operations
from me2ai_mcp.services.backend_operations import (
    # Helper functions
    match_metadata,
    match_document,
    # ChromaDB operations
    create_collection_chroma,
    upsert_chroma,
    query_chroma,
    delete_chroma,
    # FAISS operations
    create_collection_faiss,
    upsert_faiss,
    query_faiss,
    delete_faiss,
    # Qdrant operations
    create_collection_qdrant,
    upsert_qdrant,
    query_qdrant,
    delete_qdrant,
    # Pinecone operations
    upsert_pinecone,
    query_pinecone,
    delete_pinecone
)

# Import service components
from .web import WebService
from me2ai_mcp.services.base import ServiceStatus

try:
    import fastapi
    from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request, Response, Body, Query, UploadFile, File
    from fastapi.responses import JSONResponse
    import uvicorn
    import pydantic
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. Vector Store service will not function.")
    BaseModel = object  # type: ignore

# Constants
DEFAULT_PORT = 8789
DEFAULT_DIMENSION = 768
DEFAULT_DISTANCE = "cosine"
VECTOR_DB_DIR = "vectorstore"

# Configure logging
logger = logging.getLogger("me2ai-mcp-vectorstore-service")


class VectorStoreType(str, Enum):
    """Types of vector stores supported."""
    
    CHROMA = "chroma"
    FAISS = "faiss"
    QDRANT = "qdrant"
    PINECONE = "pinecone"


class EmbeddingModel(str, Enum):
    """Types of embedding models supported."""
    
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OPENAI = "openai"
    COHERE = "cohere"
    HF = "huggingface"


class UpsertRequest(BaseModel):
    """Request model for upserting documents."""
    
    documents: List[str] = Field(..., description="List of document texts to embed")
    metadatas: Optional[List[Dict[str, Any]]] = Field(None, description="Metadata for each document")
    ids: Optional[List[str]] = Field(None, description="IDs for each document")
    collection_name: Optional[str] = Field("default", description="Collection name")


class QueryRequest(BaseModel):
    """Request model for querying the vector store."""
    
    query: str = Field(..., description="Query text")
    collection_name: Optional[str] = Field("default", description="Collection name")
    n_results: Optional[int] = Field(5, description="Number of results to return")
    where: Optional[Dict[str, Any]] = Field(None, description="Metadata filter")
    where_document: Optional[Dict[str, Any]] = Field(None, description="Document content filter")


class DeleteRequest(BaseModel):
    """Request model for deleting documents."""
    
    ids: Optional[List[str]] = Field(None, description="Document IDs to delete")
    collection_name: Optional[str] = Field("default", description="Collection name")
    where: Optional[Dict[str, Any]] = Field(None, description="Metadata filter")
    where_document: Optional[Dict[str, Any]] = Field(None, description="Document content filter")


class VectorStoreService(WebService):
    """
    Microservice for vector storage and semantic search.
    
    This service provides document embedding and semantic search capabilities
    through various vector database backends.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        version: str = "0.1.0",
        store_type: VectorStoreType = VectorStoreType.CHROMA,
        embedding_type: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMERS,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Vector Store service.
        
        Args:
            host: Host to bind the service to
            port: Port to bind the service to
            version: Service version
            store_type: Type of vector store to use
            embedding_type: Type of embedding model to use
            embedding_model: Name or path of embedding model
            persist_directory: Directory to persist vector store
            metadata: Additional service metadata
        """
        # Set up metadata
        metadata = metadata or {}
        metadata.update({
            "store_type": store_type,
            "embedding_type": embedding_type,
            "embedding_model": embedding_model
        })
        
        # Initialize base web service
        super().__init__(
            name="vectorstore", 
            host=host, 
            port=port, 
            version=version,
            metadata=metadata,
            enable_cors=True,
            cors_origins=["*"],
            enable_docs=True
        )
        
        # Set up Vector Store properties
        self.store_type = store_type
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model
        
        # Set persist directory
        if persist_directory:
            self.persist_directory = persist_directory
        else:
            data_dir = os.getenv("DATA_DIR", os.path.join(tempfile.gettempdir(), "me2ai_mcp"))
            self.persist_directory = os.path.join(data_dir, VECTOR_DB_DIR)
            
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize vector store and embedding function
        self.vector_store = None
        self.embedding_function = None
        self.collections = set(["default"])
        
        # Register service endpoints
        self._register_service_endpoints()
    
    def _register_service_endpoints(self) -> None:
        """Register Vector Store service endpoints."""
        if not FASTAPI_AVAILABLE:
            self.logger.error("FastAPI is required for web services")
            return
            
        # Register upsert endpoint
        self.register_route(
            path="/upsert",
            method="POST",
            handler=self.handle_upsert,
            description="Add documents to the vector store"
        )
        
        # Register query endpoint
        self.register_route(
            path="/query",
            method="POST",
            handler=self.handle_query,
            description="Query the vector store for similar documents"
        )
        
        # Register delete endpoint
        self.register_route(
            path="/delete",
            method="POST",
            handler=self.handle_delete,
            description="Delete documents from the vector store"
        )
        
        # Register collection endpoints
        self.register_route(
            path="/collections",
            method="GET",
            handler=self.handle_list_collections,
            description="List all collections"
        )
        
        self.register_route(
            path="/collections",
            method="POST",
            handler=self.handle_create_collection,
            description="Create a new collection"
        )
        
        # Register document upload endpoint
        self.register_route(
            path="/upload",
            method="POST",
            handler=self.handle_upload,
            description="Upload a document for embedding"
        )
    
    async def start(self) -> bool:
        """
        Start the Vector Store service.
        
        Returns:
            bool: True if the service started successfully
        """
        try:
            # Initialize vector store
            await self._init_vector_store()
            
            # Start the base web service
            return await super().start()
            
        except Exception as e:
            self.logger.error(f"Error starting Vector Store service: {str(e)}")
            self.status = ServiceStatus.ERROR
            return False
    
    async def _init_vector_store(self) -> None:
        """
        Initialize the vector store and embedding function.
        
        Raises:
            ImportError: If required dependencies are not available
            ValueError: If configuration is invalid
        """
        # Import here to avoid circular imports
        try:
            # Initialize embedding function
            await self._init_embedding_function()
            
            # Initialize vector store based on type
            if self.store_type == VectorStoreType.CHROMA:
                await self._init_chroma()
            elif self.store_type == VectorStoreType.FAISS:
                await self._init_faiss()
            elif self.store_type == VectorStoreType.QDRANT:
                await self._init_qdrant()
            elif self.store_type == VectorStoreType.PINECONE:
                await self._init_pinecone()
            else:
                raise ValueError(f"Unsupported vector store type: {self.store_type}")
                
            self.logger.info(f"Initialized {self.store_type} vector store with {self.embedding_type} embeddings")
            
        except ImportError as e:
            self.logger.error(f"Missing dependencies for {self.store_type} vector store: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    async def _init_embedding_function(self) -> None:
        """
        Initialize the embedding function.
        
        Raises:
            ImportError: If required dependencies are not available
            ValueError: If configuration is invalid
        """
        if self.embedding_type == EmbeddingModel.SENTENCE_TRANSFORMERS:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.embedding_model)
                self.embedding_function = model.encode
            except ImportError:
                raise ImportError("sentence-transformers is required for SentenceTransformer embeddings")
                
        elif self.embedding_type == EmbeddingModel.OPENAI:
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embeddings")
                openai.api_key = api_key
                
                async def openai_embedding_function(texts):
                    if isinstance(texts, str):
                        texts = [texts]
                    response = await openai.Embedding.acreate(
                        input=texts,
                        model=self.embedding_model or "text-embedding-ada-002"
                    )
                    return [item["embedding"] for item in response["data"]]
                    
                self.embedding_function = openai_embedding_function
            except ImportError:
                raise ImportError("openai is required for OpenAI embeddings")
                
        elif self.embedding_type == EmbeddingModel.COHERE:
            try:
                import cohere
                api_key = os.getenv("COHERE_API_KEY")
                if not api_key:
                    raise ValueError("COHERE_API_KEY environment variable is required for Cohere embeddings")
                co = cohere.Client(api_key)
                
                def cohere_embedding_function(texts):
                    if isinstance(texts, str):
                        texts = [texts]
                    response = co.embed(
                        texts=texts,
                        model=self.embedding_model or "large"
                    )
                    return response.embeddings
                    
                self.embedding_function = cohere_embedding_function
            except ImportError:
                raise ImportError("cohere is required for Cohere embeddings")
                
        elif self.embedding_type == EmbeddingModel.HF:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
                model = AutoModel.from_pretrained(self.embedding_model)
                
                def hf_embedding_function(texts):
                    if isinstance(texts, str):
                        texts = [texts]
                    encoded_input = tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    )
                    with torch.no_grad():
                        model_output = model(**encoded_input)
                    return model_output.last_hidden_state[:, 0, :].numpy()
                    
                self.embedding_function = hf_embedding_function
            except ImportError:
                raise ImportError("transformers and torch are required for HuggingFace embeddings")
                
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")
            
    async def _init_chroma(self) -> None:
        """
        Initialize a ChromaDB vector store.
        
        Raises:
            ImportError: If required dependencies are not available
        """
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create Chroma client
            self.logger.info(f"Initializing ChromaDB with persist_directory={self.persist_directory}")
            client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            
            # Get or create default collection
            try:
                collection = client.get_collection("default")
                self.logger.info("Using existing 'default' collection")
            except Exception:
                collection = client.create_collection(
                    name="default",
                    embedding_function=self.embedding_function
                )
                self.logger.info("Created new 'default' collection")
                
            # Store client and default collection
            self.vector_store = {
                "client": client,
                "collections": {"default": collection}
            }
            
            # Get existing collections
            all_collections = client.list_collections()
            self.collections = set([coll.name for coll in all_collections])
            
        except ImportError as e:
            raise ImportError(f"chromadb is required for Chroma vector store: {str(e)}")
    
    async def _init_faiss(self) -> None:
        """
        Initialize a FAISS vector store.
        
        Raises:
            ImportError: If required dependencies are not available
        """
        try:
            import faiss
            import numpy as np
            import pickle
            
            # Create FAISS index directory
            faiss_dir = os.path.join(self.persist_directory, "faiss")
            os.makedirs(faiss_dir, exist_ok=True)
            
            # Create or load default index
            default_index_path = os.path.join(faiss_dir, "default.index")
            default_data_path = os.path.join(faiss_dir, "default.pickle")
            
            if os.path.exists(default_index_path) and os.path.exists(default_data_path):
                # Load existing index
                index = faiss.read_index(default_index_path)
                with open(default_data_path, "rb") as f:
                    data = pickle.load(f)
                self.logger.info("Loaded existing FAISS index")
            else:
                # Create new index
                dimension = DEFAULT_DIMENSION
                index = faiss.IndexFlatL2(dimension)
                data = {
                    "ids": [],
                    "texts": [],
                    "metadatas": []
                }
                self.logger.info("Created new FAISS index")
                
            # Set up collections dict
            collections = {}
            collections["default"] = {
                "index": index,
                "data": data
            }
            
            # Store FAISS client and collections
            self.vector_store = {
                "directory": faiss_dir,
                "collections": collections
            }
            
            # Get existing collections
            self.collections = set([name.split(".")[0] for name in os.listdir(faiss_dir) 
                                   if name.endswith(".index")])
            if not self.collections:
                self.collections = {"default"}
                
        except ImportError as e:
            raise ImportError(f"faiss-cpu is required for FAISS vector store: {str(e)}")
    
    async def _init_qdrant(self) -> None:
        """
        Initialize a Qdrant vector store.
        
        Raises:
            ImportError: If required dependencies are not available
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            
            # Determine if we're using local or remote Qdrant
            qdrant_url = os.getenv("QDRANT_URL")
            
            if qdrant_url:
                # Connect to remote Qdrant
                qdrant_api_key = os.getenv("QDRANT_API_KEY")
                client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key
                )
                self.logger.info(f"Connected to remote Qdrant at {qdrant_url}")
            else:
                # Use local Qdrant
                qdrant_dir = os.path.join(self.persist_directory, "qdrant")
                os.makedirs(qdrant_dir, exist_ok=True)
                client = QdrantClient(path=qdrant_dir)
                self.logger.info(f"Using local Qdrant at {qdrant_dir}")
                
            # Make sure default collection exists
            try:
                client.get_collection(collection_name="default")
                self.logger.info("Using existing 'default' collection")
            except Exception:
                # Create default collection
                client.create_collection(
                    collection_name="default",
                    vectors_config=models.VectorParams(
                        size=DEFAULT_DIMENSION,
                        distance=DEFAULT_DISTANCE
                    )
                )
                self.logger.info("Created new 'default' collection")
                
            # Store client
            self.vector_store = {
                "client": client,
                "embedding_function": self.embedding_function
            }
            
            # Get existing collections
            collections = client.get_collections().collections
            self.collections = set([coll.name for coll in collections])
            
        except ImportError as e:
            raise ImportError(f"qdrant-client is required for Qdrant vector store: {str(e)}")
    
    async def _init_pinecone(self) -> None:
        """
        Initialize a Pinecone vector store.
        
        Raises:
            ImportError: If required dependencies are not available
            ValueError: If required environment variables are not set
        """
        try:
            import pinecone
            
            # Get Pinecone API key and environment
            api_key = os.getenv("PINECONE_API_KEY")
            environment = os.getenv("PINECONE_ENVIRONMENT")
            
            if not api_key or not environment:
                raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT environment variables are required")
                
            # Initialize Pinecone
            pinecone.init(api_key=api_key, environment=environment)
            
            # Get or create default index
            index_name = os.getenv("PINECONE_INDEX", "me2ai-mcp")
            
            # Check if index exists
            indexes = pinecone.list_indexes()
            
            if index_name not in indexes:
                # Create index
                pinecone.create_index(
                    name=index_name,
                    dimension=DEFAULT_DIMENSION,
                    metric=DEFAULT_DISTANCE
                )
                self.logger.info(f"Created new Pinecone index '{index_name}'")
            else:
                self.logger.info(f"Using existing Pinecone index '{index_name}'")
                
            # Get index
            index = pinecone.Index(index_name)
            
            # Store client
            self.vector_store = {
                "client": index,
                "embedding_function": self.embedding_function
            }
            
            # For Pinecone, we use namespaces instead of collections
            # Get stats to see namespaces
            stats = index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            
            # Add default namespace if none exist
            if not namespaces:
                namespaces = {"default": {"vector_count": 0}}
                
            self.collections = set(namespaces.keys())
            
        except ImportError as e:
            raise ImportError(f"pinecone-client is required for Pinecone vector store: {str(e)}")
    
    async def handle_upsert(self, request: Request, params: UpsertRequest) -> Dict[str, Any]:
        """
        Handle upsert request to add documents to the vector store.
        
        Args:
            request: FastAPI request object
            params: Upsert parameters
            
        Returns:
            Dict[str, Any]: Upsert results
            
        Raises:
            HTTPException: If the request fails
        """
        try:
            documents = params.documents
            metadatas = params.metadatas or [{}] * len(documents)
            ids = params.ids or [str(uuid.uuid4()) for _ in range(len(documents))]
            collection_name = params.collection_name or "default"
            
            # Validate inputs
            if len(documents) != len(metadatas) or len(documents) != len(ids):
                raise HTTPException(
                    status_code=400,
                    detail="Number of documents, metadatas, and IDs must match"
                )
                
            # Check if collection exists
            if collection_name not in self.collections:
                await self._create_collection(collection_name)
                
            # Upsert documents based on vector store type
            if self.store_type == VectorStoreType.CHROMA:
                await upsert_chroma(
                    self.vector_store["client"],
                    self.vector_store["collections"],
                    collection_name,
                    documents,
                    metadatas,
                    ids,
                    self.embedding_function
                )
            elif self.store_type == VectorStoreType.FAISS:
                await upsert_faiss(
                    self.vector_store["directory"],
                    self.vector_store["collections"],
                    collection_name,
                    documents,
                    metadatas,
                    ids,
                    self.embedding_function
                )
            elif self.store_type == VectorStoreType.QDRANT:
                await upsert_qdrant(
                    self.vector_store["client"],
                    collection_name,
                    documents,
                    metadatas,
                    ids,
                    self.embedding_function
                )
            elif self.store_type == VectorStoreType.PINECONE:
                await upsert_pinecone(
                    self.vector_store["client"],
                    collection_name,
                    documents,
                    metadatas,
                    ids,
                    self.embedding_function
                )
            else:
                raise HTTPException(
                    status_code=501,
                    detail=f"Unsupported vector store type: {self.store_type}"
                )
                
            return {
                "success": True,
                "ids": ids,
                "collection": collection_name,
                "count": len(documents)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error upserting documents: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error upserting documents: {str(e)}"
            )
    
    async def handle_query(self, request: Request, params: QueryRequest) -> Dict[str, Any]:
        """
        Handle query request to search for similar documents.
        
        Args:
            request: FastAPI request object
            params: Query parameters
            
        Returns:
            Dict[str, Any]: Query results
            
        Raises:
            HTTPException: If the request fails
        """
        try:
            query = params.query
            collection_name = params.collection_name or "default"
            n_results = params.n_results or 5
            where = params.where
            where_document = params.where_document
            
            # Check if collection exists
            if collection_name not in self.collections:
                raise HTTPException(
                    status_code=404,
                    detail=f"Collection not found: {collection_name}"
                )
                
            # Query documents based on vector store type
            if self.store_type == VectorStoreType.CHROMA:
                results = await query_chroma(
                    self.vector_store["collections"],
                    collection_name,
                    query,
                    n_results,
                    where,
                    where_document
                )
            elif self.store_type == VectorStoreType.FAISS:
                results = await query_faiss(
                    self.vector_store["collections"],
                    collection_name,
                    query,
                    n_results,
                    where,
                    where_document,
                    self.embedding_function
                )
            elif self.store_type == VectorStoreType.QDRANT:
                results = await query_qdrant(
                    self.vector_store["client"],
                    collection_name,
                    query,
                    n_results,
                    where,
                    where_document,
                    self.embedding_function
                )
            elif self.store_type == VectorStoreType.PINECONE:
                results = await query_pinecone(
                    self.vector_store["client"],
                    collection_name,
                    query,
                    n_results,
                    where,
                    where_document,
                    self.embedding_function
                )
            else:
                raise HTTPException(
                    status_code=501,
                    detail=f"Unsupported vector store type: {self.store_type}"
                )
                
            return {
                "success": True,
                "query": query,
                "collection": collection_name,
                "results": results
            }
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error querying documents: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error querying documents: {str(e)}"
            )
    
    async def handle_delete(self, request: Request, params: DeleteRequest) -> Dict[str, Any]:
        """
        Handle delete request to remove documents from the vector store.
        
        Args:
            request: FastAPI request object
            params: Delete parameters
            
        Returns:
            Dict[str, Any]: Delete results
            
        Raises:
            HTTPException: If the request fails
        """
        try:
            ids = params.ids
            collection_name = params.collection_name or "default"
            where = params.where
            where_document = params.where_document
            
            # Check if collection exists
            if collection_name not in self.collections:
                raise HTTPException(
                    status_code=404,
                    detail=f"Collection not found: {collection_name}"
                )
                
            # Delete documents based on vector store type
            if self.store_type == VectorStoreType.CHROMA:
                count = await delete_chroma(
                    self.vector_store["collections"],
                    collection_name,
                    ids,
                    where,
                    where_document
                )
            elif self.store_type == VectorStoreType.FAISS:
                count = await delete_faiss(
                    self.vector_store["directory"],
                    self.vector_store["collections"],
                    collection_name,
                    ids,
                    where,
                    where_document,
                    self.embedding_function
                )
            elif self.store_type == VectorStoreType.QDRANT:
                count = await delete_qdrant(
                    self.vector_store["client"],
                    collection_name,
                    ids,
                    where,
                    where_document
                )
            elif self.store_type == VectorStoreType.PINECONE:
                count = await delete_pinecone(
                    self.vector_store["client"],
                    collection_name,
                    ids,
                    where,
                    where_document
                )
            else:
                raise HTTPException(
                    status_code=501,
                    detail=f"Unsupported vector store type: {self.store_type}"
                )
                
            return {
                "success": True,
                "collection": collection_name,
                "deleted_count": count
            }
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error deleting documents: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error deleting documents: {str(e)}"
            )
    
    async def handle_list_collections(self, request: Request) -> Dict[str, Any]:
        """
        Handle request to list all collections.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Dict[str, Any]: List of collections
        """
        try:
            return {
                "success": True,
                "collections": list(self.collections)
            }
        except Exception as e:
            self.logger.error(f"Error listing collections: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error listing collections: {str(e)}"
            )
    
    async def handle_create_collection(self, request: Request, name: str = Body(..., embed=True)) -> Dict[str, Any]:
        """
        Handle request to create a new collection.
        
        Args:
            request: FastAPI request object
            name: Name of the collection to create
            
        Returns:
            Dict[str, Any]: Creation result
            
        Raises:
            HTTPException: If the request fails
        """
        try:
            # Check if collection already exists
            if name in self.collections:
                return {
                    "success": True,
                    "message": f"Collection already exists: {name}",
                    "collection": name
                }
                
            # Create collection
            await self._create_collection(name)
            
            return {
                "success": True,
                "message": f"Collection created: {name}",
                "collection": name
            }
            
        except Exception as e:
            self.logger.error(f"Error creating collection: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error creating collection: {str(e)}"
            )
    
    async def handle_upload(self, request: Request, file: UploadFile = File(...), collection: str = "default") -> Dict[str, Any]:
        """
        Handle request to upload and embed a document file.
        
        Args:
            request: FastAPI request object
            file: File to upload
            collection: Collection to add document to
            
        Returns:
            Dict[str, Any]: Upload result
            
        Raises:
            HTTPException: If the request fails
        """
        try:
            # Check if collection exists
            if collection not in self.collections:
                await self._create_collection(collection)
                
            # Read file content
            content = await file.read()
            text = content.decode("utf-8", errors="ignore")
            
            # Add max content length check
            max_length = 100000  # 100KB text limit
            if len(text) > max_length:
                self.logger.warning(f"Truncating content from {len(text)} to {max_length} characters")
                text = text[:max_length]
            
            # Basic metadata
            metadata = {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(content),
                "uploaded_at": time.time()
            }
            
            # Generate ID
            doc_id = str(uuid.uuid4())
            
            # Upsert document based on vector store type
            if self.store_type == VectorStoreType.CHROMA:
                await upsert_chroma(
                    self.vector_store["client"],
                    self.vector_store["collections"],
                    collection,
                    [text],
                    [metadata],
                    [doc_id],
                    self.embedding_function
                )
            elif self.store_type == VectorStoreType.FAISS:
                await upsert_faiss(
                    self.vector_store["directory"],
                    self.vector_store["collections"],
                    collection,
                    [text],
                    [metadata],
                    [doc_id],
                    self.embedding_function
                )
            elif self.store_type == VectorStoreType.QDRANT:
                await upsert_qdrant(
                    self.vector_store["client"],
                    collection,
                    [text],
                    [metadata],
                    [doc_id],
                    self.embedding_function
                )
            elif self.store_type == VectorStoreType.PINECONE:
                await upsert_pinecone(
                    self.vector_store["client"],
                    collection,
                    [text],
                    [metadata],
                    [doc_id],
                    self.embedding_function
                )
            else:
                raise HTTPException(
                    status_code=501,
                    detail=f"Unsupported vector store type: {self.store_type}"
                )
                
            return {
                "success": True,
                "id": doc_id,
                "filename": file.filename,
                "collection": collection,
                "chars": len(text)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error uploading document: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error uploading document: {str(e)}"
            )
            
    async def _create_collection(self, name: str) -> None:
        """
        Create a new collection in the vector store.
        
        Args:
            name: Name of the collection to create
            
        Raises:
            Exception: If creation fails
        """
        try:
            if self.store_type == VectorStoreType.CHROMA:
                collection = await create_collection_chroma(
                    self.vector_store["client"],
                    name,
                    self.embedding_function
                )
                self.vector_store["collections"][name] = collection
            elif self.store_type == VectorStoreType.FAISS:
                collection = await create_collection_faiss(
                    self.vector_store["directory"],
                    name
                )
                self.vector_store["collections"][name] = collection
            elif self.store_type == VectorStoreType.QDRANT:
                await create_collection_qdrant(
                    self.vector_store["client"],
                    name
                )
            elif self.store_type == VectorStoreType.PINECONE:
                # For Pinecone, we just use namespaces which don't need explicit creation
                pass
            else:
                raise ValueError(f"Unsupported vector store type: {self.store_type}")
                
            # Add to collections set
            self.collections.add(name)
            self.logger.info(f"Created collection: {name}")
            
        except Exception as e:
            self.logger.error(f"Error creating collection {name}: {str(e)}")
            raise
