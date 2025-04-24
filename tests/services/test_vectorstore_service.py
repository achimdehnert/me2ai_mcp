"""
Tests for VectorStore service.

This module tests the VectorStore service for ME2AI MCP,
validating its embedding and semantic search capabilities.
"""

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
import json
import tempfile
import shutil
import uuid
import pytest
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple

# Import the service under test
from me2ai_mcp.services.vectorstore_service import (
    VectorStoreService,
    VectorStoreType,
    EmbeddingModel,
    UpsertRequest,
    QueryRequest,
    DeleteRequest
)


class TestVectorStoreService(unittest.TestCase):
    """Test suite for VectorStore service.
    
    This comprehensive test suite validates the functionality of the VectorStore
    service with various vector stores and embedding models. It tests initialization,
    lifecycle, API endpoints, error handling, and edge cases.
    """
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create temp directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Mock embedding function
        self.mock_embedding_function = MagicMock()
        self.mock_embedding_function.return_value = [0.1] * 768  # Default dimension
        
        # Set mock values
        self.service_name = "vectorstore"
        self.host = "localhost"
        self.test_port = 9876
        self.test_version = "0.1.0-test"
        
        # Default test documents
        self.test_docs = [
            "This is a test document about AI and machine learning.",
            "Vector databases store embeddings for semantic search.",
            "ME2AI MCP provides tools for knowledge management."
        ]
        self.test_ids = [str(uuid.uuid4()) for _ in range(len(self.test_docs))]
        self.test_metadata = [
            {"source": "test", "category": "ai"},
            {"source": "test", "category": "database"},
            {"source": "test", "category": "framework"}
        ]
            
    def tearDown(self) -> None:
        """Tear down test fixtures."""
        # Clean up temp directory
        shutil.rmtree(self.test_dir)
        
        # Clean up any running services
        if hasattr(self, "service") and self.service:
            asyncio.run(self.service.stop())
            
    @patch("me2ai_mcp.services.vectorstore_service.FASTAPI_AVAILABLE", True)
    @patch("me2ai_mcp.services.vectorstore_service.SentenceTransformer")
    async def test_service_init(self, mock_sentence_transformer: MagicMock) -> None:
        """Test service initialization with various vector store types.
        
        This test validates that the service initializes correctly with default
        parameters and different vector store types. It ensures that all service
        properties are set correctly and all required endpoints are registered.
        """
        # Set up the mock
        mock_model = MagicMock()
        mock_model.encode = self.mock_embedding_function
        mock_sentence_transformer.return_value = mock_model
        
        # Test with ChromaDB (default)
        with patch("me2ai_mcp.services.vectorstore_service.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.create_collection.return_value = mock_collection
            mock_client.list_collections.return_value = []
            mock_chromadb.PersistentClient.return_value = mock_client
            
            service = VectorStoreService(
                host=self.host,
                port=self.test_port,
                version=self.test_version,
                persist_directory=self.test_dir
            )
            
            # Verify service properties
            self.assertEqual(service.name, self.service_name)
            self.assertEqual(service.host, self.host)
            self.assertEqual(service.port, self.test_port)
            self.assertEqual(service.version, self.test_version)
            self.assertEqual(service.store_type, VectorStoreType.CHROMA)
            self.assertEqual(service.embedding_type, EmbeddingModel.SENTENCE_TRANSFORMERS)
            self.assertEqual(service.embedding_model, "all-MiniLM-L6-v2")
            self.assertEqual(service.persist_directory, self.test_dir)
            
            # Verify endpoints registered
            self.assertTrue(len(service.endpoints) > 0)
            endpoint_paths = [ep.path for ep in service.endpoints]
            self.assertIn("/upsert", endpoint_paths)
            self.assertIn("/query", endpoint_paths)
            self.assertIn("/delete", endpoint_paths)
            self.assertIn("/collections", endpoint_paths)
            self.assertIn("/upload", endpoint_paths)
            
        # Test with FAISS
        with patch("me2ai_mcp.services.vectorstore_service.faiss") as mock_faiss:
            mock_faiss.IndexFlatL2.return_value = MagicMock()
            
            service = VectorStoreService(
                host=self.host,
                port=self.test_port,
                version=self.test_version,
                store_type=VectorStoreType.FAISS,
                persist_directory=self.test_dir
            )
            
            self.assertEqual(service.store_type, VectorStoreType.FAISS)
                
        # Test with other embedding models
        with patch("me2ai_mcp.services.vectorstore_service.openai") as mock_openai:
            # Set environment variable for testing
            os.environ["OPENAI_API_KEY"] = "test_key"
            
            service = VectorStoreService(
                host=self.host,
                port=self.test_port,
                version=self.test_version,
                embedding_type=EmbeddingModel.OPENAI,
                persist_directory=self.test_dir
            )
            
            self.assertEqual(service.embedding_type, EmbeddingModel.OPENAI)
            
    @patch("me2ai_mcp.services.vectorstore_service.FASTAPI_AVAILABLE", True)
    @patch("me2ai_mcp.services.vectorstore_service.SentenceTransformer")
    @patch("me2ai_mcp.services.vectorstore_service.chromadb")
    async def test_service_lifecycle(
        self, 
        mock_chromadb: MagicMock, 
        mock_sentence_transformer: MagicMock
    ) -> None:
        """Test service start and stop lifecycle."""
        # Set up the mocks
        mock_model = MagicMock()
        mock_model.encode = self.mock_embedding_function
        mock_sentence_transformer.return_value = mock_model
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_client.list_collections.return_value = []
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Create service
        service = VectorStoreService(
            host=self.host,
            port=self.test_port,
            version=self.test_version,
            persist_directory=self.test_dir
        )
        
        # Mock uvicorn
        with patch("me2ai_mcp.services.web.uvicorn") as mock_uvicorn:
            mock_uvicorn.Config = MagicMock()
            mock_uvicorn.Server = MagicMock()
            mock_server = mock_uvicorn.Server.return_value
            mock_server.serve = AsyncMock()
            
            # Start the service
            start_result = await service.start()
            self.assertTrue(start_result)
            
            # Verify uvicorn was called
            mock_uvicorn.Config.assert_called_once()
            mock_server.serve.assert_called_once()
            
            # Stop the service
            stop_result = await service.stop()
            self.assertTrue(stop_result)
            
    @patch("me2ai_mcp.services.vectorstore_service.FASTAPI_AVAILABLE", True)
    @patch("me2ai_mcp.services.vectorstore_service.SentenceTransformer")
    @patch("me2ai_mcp.services.vectorstore_service.chromadb")  
    async def test_handle_upsert(
        self, 
        mock_chromadb: MagicMock, 
        mock_sentence_transformer: MagicMock
    ) -> None:
        """Test upserting documents to the vector store."""
        # Set up the mocks
        mock_model = MagicMock()
        mock_model.encode = self.mock_embedding_function
        mock_sentence_transformer.return_value = mock_model
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.return_value = mock_collection
        mock_client.list_collections.return_value = []
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Create service
        service = VectorStoreService(
            host=self.host,
            port=self.test_port,
            version=self.test_version,
            persist_directory=self.test_dir
        )
        
        # Initialize service components
        await service._init_vector_store()
        
        # Create mock request
        mock_request = MagicMock()
        
        # Create valid upsert request
        upsert_params = UpsertRequest(
            documents=self.test_docs,
            metadatas=self.test_metadata,
            ids=self.test_ids,
            collection_name="test_collection"
        )
        
        # Patch the upsert_chroma function
        with patch("me2ai_mcp.services.vectorstore_service.upsert_chroma", new=AsyncMock()) as mock_upsert:
            # Call the handler
            result = await service.handle_upsert(mock_request, upsert_params)
            
            # Verify result format
            self.assertIn("success", result)
            self.assertTrue(result["success"])
            self.assertIn("ids", result)
            self.assertEqual(result["ids"], self.test_ids)
            self.assertIn("collection", result)
            self.assertEqual(result["collection"], "test_collection")
            self.assertIn("count", result)
            self.assertEqual(result["count"], len(self.test_docs))
            
            # Verify upsert was called
            mock_upsert.assert_called_once()
            
    @patch("me2ai_mcp.services.vectorstore_service.FASTAPI_AVAILABLE", True)
    @patch("me2ai_mcp.services.vectorstore_service.SentenceTransformer")
    @patch("me2ai_mcp.services.vectorstore_service.chromadb")
    async def test_handle_query(
        self, 
        mock_chromadb: MagicMock, 
        mock_sentence_transformer: MagicMock
    ) -> None:
        """Test querying the vector store."""
        # Set up the mocks
        mock_model = MagicMock()
        mock_model.encode = self.mock_embedding_function
        mock_sentence_transformer.return_value = mock_model
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.return_value = mock_collection
        mock_client.list_collections.return_value = []
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Create service
        service = VectorStoreService(
            host=self.host,
            port=self.test_port,
            version=self.test_version,
            persist_directory=self.test_dir
        )
        
        # Initialize service components
        await service._init_vector_store()
        service.collections.add("test_collection")
        
        # Create mock request
        mock_request = MagicMock()
        
        # Create valid query request
        query_params = QueryRequest(
            query="What is vector search?",
            collection_name="test_collection",
            n_results=3
        )
        
        # Mock query results
        mock_results = [
            {
                "id": self.test_ids[1],
                "text": self.test_docs[1],
                "metadata": self.test_metadata[1],
                "distance": 0.1
            },
            {
                "id": self.test_ids[0],
                "text": self.test_docs[0],
                "metadata": self.test_metadata[0],
                "distance": 0.2
            }
        ]
        
        # Patch the query_chroma function
        with patch("me2ai_mcp.services.vectorstore_service.query_chroma", new=AsyncMock(return_value=mock_results)) as mock_query:
            # Call the handler
            result = await service.handle_query(mock_request, query_params)
            
            # Verify result format
            self.assertIn("success", result)
            self.assertTrue(result["success"])
            self.assertIn("query", result)
            self.assertEqual(result["query"], "What is vector search?")
            self.assertIn("collection", result)
            self.assertEqual(result["collection"], "test_collection")
            self.assertIn("results", result)
            self.assertEqual(result["results"], mock_results)
            
            # Verify query was called
            mock_query.assert_called_once()
            
    @patch("me2ai_mcp.services.vectorstore_service.FASTAPI_AVAILABLE", True)
    @patch("me2ai_mcp.services.vectorstore_service.SentenceTransformer")
    @patch("me2ai_mcp.services.vectorstore_service.chromadb")
    async def test_handle_delete(
        self, 
        mock_chromadb: MagicMock, 
        mock_sentence_transformer: MagicMock
    ) -> None:
        """Test deleting documents from the vector store."""
        # Set up the mocks
        mock_model = MagicMock()
        mock_model.encode = self.mock_embedding_function
        mock_sentence_transformer.return_value = mock_model
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.return_value = mock_collection
        mock_client.list_collections.return_value = []
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Create service
        service = VectorStoreService(
            host=self.host,
            port=self.test_port,
            version=self.test_version,
            persist_directory=self.test_dir
        )
        
        # Initialize service components
        await service._init_vector_store()
        service.collections.add("test_collection")
        
        # Create mock request
        mock_request = MagicMock()
        
        # Create valid delete request
        delete_params = DeleteRequest(
            ids=[self.test_ids[0]],
            collection_name="test_collection"
        )
        
        # Patch the delete_chroma function
        with patch("me2ai_mcp.services.vectorstore_service.delete_chroma", new=AsyncMock(return_value=1)) as mock_delete:
            # Call the handler
            result = await service.handle_delete(mock_request, delete_params)
            
            # Verify result format
            self.assertIn("success", result)
            self.assertTrue(result["success"])
            self.assertIn("collection", result)
            self.assertEqual(result["collection"], "test_collection")
            self.assertIn("deleted_count", result)
            self.assertEqual(result["deleted_count"], 1)
            
            # Verify delete was called
            mock_delete.assert_called_once()
            
    @patch("me2ai_mcp.services.vectorstore_service.FASTAPI_AVAILABLE", True)
    @patch("me2ai_mcp.services.vectorstore_service.SentenceTransformer")
    @patch("me2ai_mcp.services.vectorstore_service.chromadb")
    async def test_collection_management(
        self, 
        mock_chromadb: MagicMock, 
        mock_sentence_transformer: MagicMock
    ) -> None:
        """Test collection management (list, create)."""
        # Set up the mocks
        mock_model = MagicMock()
        mock_model.encode = self.mock_embedding_function
        mock_sentence_transformer.return_value = mock_model
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.return_value = mock_collection
        mock_client.list_collections.return_value = []
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Create service
        service = VectorStoreService(
            host=self.host,
            port=self.test_port,
            version=self.test_version,
            persist_directory=self.test_dir
        )
        
        # Initialize service components
        await service._init_vector_store()
        
        # Add a test collection
        service.collections.add("test_collection")
        
        # Create mock request
        mock_request = MagicMock()
        
        # Test listing collections
        list_result = await service.handle_list_collections(mock_request)
        self.assertIn("success", list_result)
        self.assertTrue(list_result["success"])
        self.assertIn("collections", list_result)
        self.assertIn("test_collection", list_result["collections"])
        
        # Patch the create_collection_chroma function
        with patch("me2ai_mcp.services.vectorstore_service.create_collection_chroma", new=AsyncMock()) as mock_create:
            # Test creating a new collection
            create_result = await service.handle_create_collection(mock_request, "new_collection")
            self.assertIn("success", create_result)
            self.assertTrue(create_result["success"])
            self.assertIn("collection", create_result)
            self.assertEqual(create_result["collection"], "new_collection")
            
            # Verify create was called
            mock_create.assert_called_once()
            
    @patch("me2ai_mcp.services.vectorstore_service.FASTAPI_AVAILABLE", True)
    @patch("me2ai_mcp.services.vectorstore_service.SentenceTransformer")
    @patch("me2ai_mcp.services.vectorstore_service.chromadb")
    async def test_handle_upload(
        self, 
        mock_chromadb: MagicMock, 
        mock_sentence_transformer: MagicMock
    ) -> None:
        """Test uploading a document to the vector store."""
        # Set up the mocks
        mock_model = MagicMock()
        mock_model.encode = self.mock_embedding_function
        mock_sentence_transformer.return_value = mock_model
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.return_value = mock_collection
        mock_client.list_collections.return_value = []
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Create service
        service = VectorStoreService(
            host=self.host,
            port=self.test_port,
            version=self.test_version,
            persist_directory=self.test_dir
        )
        
        # Initialize service components
        await service._init_vector_store()
        service.collections.add("test_collection")
        
        # Create mock request
        mock_request = MagicMock()
        
        # Create mock file
        mock_file = MagicMock()
        mock_file.filename = "test.txt"
        mock_file.content_type = "text/plain"
        mock_file.read = AsyncMock(return_value=b"This is a test document for upload.")
        
        # Patch the upsert_chroma function
        with patch("me2ai_mcp.services.vectorstore_service.upsert_chroma", new=AsyncMock()) as mock_upsert:
            # Call the handler
            result = await service.handle_upload(mock_request, mock_file, "test_collection")
            
            # Verify result format
            self.assertIn("success", result)
            self.assertTrue(result["success"])
            self.assertIn("id", result)
            self.assertIn("filename", result)
            self.assertEqual(result["filename"], "test.txt")
            self.assertIn("collection", result)
            self.assertEqual(result["collection"], "test_collection")
            self.assertIn("chars", result)
            self.assertEqual(result["chars"], len("This is a test document for upload."))
            
            # Verify upsert was called
            mock_upsert.assert_called_once()
            
    @patch("me2ai_mcp.services.vectorstore_service.FASTAPI_AVAILABLE", True)
    @patch("me2ai_mcp.services.vectorstore_service.SentenceTransformer")
    async def test_different_vector_stores(
        self,
        mock_sentence_transformer: MagicMock
    ) -> None:
        """Test initialization with different vector store types."""
        # Set up the mock
        mock_model = MagicMock()
        mock_model.encode = self.mock_embedding_function
        mock_sentence_transformer.return_value = mock_model
        
        # Test FAISS
        with patch("me2ai_mcp.services.vectorstore_service.faiss") as mock_faiss:
            mock_faiss.IndexFlatL2.return_value = MagicMock()
            
            service = VectorStoreService(
                host=self.host,
                port=self.test_port,
                version=self.test_version,
                store_type=VectorStoreType.FAISS,
                persist_directory=self.test_dir
            )
            
            self.assertEqual(service.store_type, VectorStoreType.FAISS)
            
        # Test Qdrant
        with patch("me2ai_mcp.services.vectorstore_service.QdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock(collections=[])
            mock_qdrant.return_value = mock_client
            
            service = VectorStoreService(
                host=self.host,
                port=self.test_port,
                version=self.test_version,
                store_type=VectorStoreType.QDRANT,
                persist_directory=self.test_dir
            )
            
            self.assertEqual(service.store_type, VectorStoreType.QDRANT)
            
        # Test Pinecone
        with patch("me2ai_mcp.services.vectorstore_service.pinecone") as mock_pinecone:
            os.environ["PINECONE_API_KEY"] = "test_key"
            os.environ["PINECONE_ENVIRONMENT"] = "test-env"
            
            mock_index = MagicMock()
            mock_index.describe_index_stats.return_value = {"namespaces": {"default": {"vector_count": 0}}}
            mock_pinecone.Index.return_value = mock_index
            mock_pinecone.list_indexes.return_value = ["me2ai-mcp"]
            
            service = VectorStoreService(
                host=self.host,
                port=self.test_port,
                version=self.test_version,
                store_type=VectorStoreType.PINECONE,
                persist_directory=self.test_dir
            )
            
            self.assertEqual(service.store_type, VectorStoreType.PINECONE)
            

# Run tests with pytest
    @patch("me2ai_mcp.services.vectorstore_service.FASTAPI_AVAILABLE", True)
    @patch("me2ai_mcp.services.vectorstore_service.SentenceTransformer")
    async def test_error_handling(self, mock_sentence_transformer: MagicMock) -> None:
        """Test error handling in VectorStore service.
        
        This test validates how the service handles various error scenarios such as:
        - Missing collections
        - Invalid queries
        - Failed operations
        - Dependency errors
        """
        # Set up the mock
        mock_model = MagicMock()
        mock_model.encode = self.mock_embedding_function
        mock_sentence_transformer.return_value = mock_model
        
        # Test with ChromaDB (default)
        with patch("me2ai_mcp.services.vectorstore_service.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.create_collection.return_value = mock_collection
            mock_client.list_collections.return_value = []
            mock_chromadb.PersistentClient.return_value = mock_client
            
            service = VectorStoreService(
                host=self.host,
                port=self.test_port,
                version=self.test_version,
                persist_directory=self.test_dir
            )
            
            # Initialize service components
            await service._init_vector_store()
            
            # Create mock request
            mock_request = MagicMock()
            
            # Test query with non-existent collection
            query_params = QueryRequest(
                query="What is vector search?",
                collection_name="nonexistent_collection",
                n_results=3
            )
            
            # Expect HTTPException for nonexistent collection
            with pytest.raises(Exception):
                await service.handle_query(mock_request, query_params)
                
            # Test delete with non-existent collection
            delete_params = DeleteRequest(
                ids=[self.test_ids[0]],
                collection_name="nonexistent_collection"
            )
            
            # Expect HTTPException for nonexistent collection
            with pytest.raises(Exception):
                await service.handle_delete(mock_request, delete_params)
                
            # Test upsert with incorrect data (different lengths)
            with pytest.raises(Exception):
                await service.handle_upsert(mock_request, UpsertRequest(
                    documents=["doc1", "doc2"],
                    metadatas=[{"source": "test"}],  # Only one metadata for two docs
                    ids=["id1", "id2"],
                    collection_name="test_collection"
                ))
                
    @patch("me2ai_mcp.services.vectorstore_service.FASTAPI_AVAILABLE", True)
    @patch("me2ai_mcp.services.vectorstore_service.SentenceTransformer")
    async def test_collection_reuse(self, mock_sentence_transformer: MagicMock) -> None:
        """Test collection reuse in VectorStore service.
        
        This test validates that existing collections are properly reused
        rather than recreated, and that state is maintained across operations.
        """
        # Set up the mock
        mock_model = MagicMock()
        mock_model.encode = self.mock_embedding_function
        mock_sentence_transformer.return_value = mock_model
        
        # Test with ChromaDB (default)
        with patch("me2ai_mcp.services.vectorstore_service.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.name = "existing_collection"
            mock_client.create_collection.return_value = mock_collection
            mock_client.get_collection.return_value = mock_collection
            mock_client.list_collections.return_value = [mock_collection]
            mock_chromadb.PersistentClient.return_value = mock_client
            
            service = VectorStoreService(
                host=self.host,
                port=self.test_port,
                version=self.test_version,
                persist_directory=self.test_dir
            )
            
            # Initialize service components
            await service._init_vector_store()
            
            # Verify existing collection is in the set
            self.assertIn("existing_collection", service.collections)
            
            # Create mock request
            mock_request = MagicMock()
            
            # Test creating an existing collection
            create_result = await service.handle_create_collection(mock_request, "existing_collection")
            self.assertIn("success", create_result)
            self.assertTrue(create_result["success"])
            self.assertIn("already exists", create_result["message"].lower())
            
            # Mock upsert to verify collection reuse
            with patch("me2ai_mcp.services.vectorstore_service.upsert_chroma", new=AsyncMock()) as mock_upsert:
                # Upsert to existing collection
                await service.handle_upsert(mock_request, UpsertRequest(
                    documents=["test document"],
                    collection_name="existing_collection"
                ))
                
                # Verify get_collection was called rather than create_collection
                mock_client.get_collection.assert_called_with("existing_collection")
                mock_upsert.assert_called_once()
                
    @patch("me2ai_mcp.services.vectorstore_service.FASTAPI_AVAILABLE", True)
    @patch("me2ai_mcp.services.vectorstore_service.SentenceTransformer")
    async def test_large_content_handling(self, mock_sentence_transformer: MagicMock) -> None:
        """Test handling of large content in VectorStore service.
        
        This test validates that large document content is properly handled,
        including truncation when necessary.
        """
        # Set up the mock
        mock_model = MagicMock()
        mock_model.encode = self.mock_embedding_function
        mock_sentence_transformer.return_value = mock_model
        
        # Test with ChromaDB (default)
        with patch("me2ai_mcp.services.vectorstore_service.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.create_collection.return_value = mock_collection
            mock_client.list_collections.return_value = []
            mock_chromadb.PersistentClient.return_value = mock_client
            
            service = VectorStoreService(
                host=self.host,
                port=self.test_port,
                version=self.test_version,
                persist_directory=self.test_dir
            )
            
            # Initialize service components
            await service._init_vector_store()
            service.collections.add("test_collection")
            
            # Create mock request
            mock_request = MagicMock()
            
            # Create mock file with very large content
            mock_file = MagicMock()
            mock_file.filename = "large_test.txt"
            mock_file.content_type = "text/plain"
            # Create text larger than the 100KB limit
            large_text = "This is a test. " * 10000  # Approx 150KB
            mock_file.read = AsyncMock(return_value=large_text.encode('utf-8'))
            
            # Patch the upsert_chroma function
            with patch("me2ai_mcp.services.vectorstore_service.upsert_chroma", new=AsyncMock()) as mock_upsert:
                # Call the handler
                result = await service.handle_upload(mock_request, mock_file, "test_collection")
                
                # Verify result format
                self.assertIn("success", result)
                self.assertTrue(result["success"])
                self.assertIn("chars", result)
                # Verify content was truncated
                self.assertEqual(result["chars"], 100000)  # Max length from code
                
                # Verify upsert was called with truncated text
                mock_upsert.assert_called_once()
                # Get the text argument from the call
                args, kwargs = mock_upsert.call_args
                # The text is the first item in the fourth argument (documents)
                self.assertEqual(len(args[3][0]), 100000)
                
    @patch("me2ai_mcp.services.vectorstore_service.FASTAPI_AVAILABLE", True)
    @patch("me2ai_mcp.services.vectorstore_service.SentenceTransformer")
    async def test_missing_dependencies(self, mock_sentence_transformer: MagicMock) -> None:
        """Test handling of missing dependencies in VectorStore service.
        
        This test validates that the service gracefully handles missing optional
        dependencies for various vector store backends.
        """
        # Set up the mock
        mock_model = MagicMock()
        mock_model.encode = self.mock_embedding_function
        mock_sentence_transformer.return_value = mock_model
        
        # Test with missing Qdrant dependency
        with patch("me2ai_mcp.services.vectorstore_service.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.create_collection.return_value = mock_collection
            mock_client.list_collections.return_value = []
            mock_chromadb.PersistentClient.return_value = mock_client
            
            # Create service with Qdrant store type
            with patch("me2ai_mcp.services.vectorstore_service._init_qdrant", side_effect=ImportError("qdrant_client not found")):
                # This should fallback to default ChromaDB
                with pytest.raises(ImportError):
                    service = VectorStoreService(
                        host=self.host,
                        port=self.test_port,
                        version=self.test_version,
                        store_type=VectorStoreType.QDRANT,
                        persist_directory=self.test_dir
                    )
                    await service._init_vector_store()
                    
        # Test with missing sentence_transformers dependency
        with patch("me2ai_mcp.services.vectorstore_service.SentenceTransformer", side_effect=ImportError("sentence_transformers not found")):
            # Test with OpenAI fallback
            with patch("me2ai_mcp.services.vectorstore_service.openai") as mock_openai:
                # Set environment variable for testing
                os.environ["OPENAI_API_KEY"] = "test_key"
                
                with patch("me2ai_mcp.services.vectorstore_service.chromadb") as mock_chromadb:
                    mock_client = MagicMock()
                    mock_collection = MagicMock()
                    mock_client.create_collection.return_value = mock_collection
                    mock_client.list_collections.return_value = []
                    mock_chromadb.PersistentClient.return_value = mock_client
                    
                    service = VectorStoreService(
                        host=self.host,
                        port=self.test_port,
                        version=self.test_version,
                        embedding_type=EmbeddingModel.OPENAI,
                        persist_directory=self.test_dir
                    )
                    
                    # This should use OpenAI embeddings instead
                    await service._init_embedding_function()
                    self.assertEqual(service.embedding_type, EmbeddingModel.OPENAI)

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
