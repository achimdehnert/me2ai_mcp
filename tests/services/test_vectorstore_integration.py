"""
Integration tests for VectorStore service.

This module contains end-to-end tests for the VectorStore service
with all supported backends, validating the complete service flow.
"""

import unittest
import asyncio
import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional
import numpy as np
import uuid

from fastapi.testclient import TestClient
from fastapi import UploadFile
from io import BytesIO

from me2ai_mcp.services.vectorstore_service import (
    VectorStoreService,
    VectorStoreType,
    EmbeddingModel
)


class TestVectorStoreIntegration(unittest.TestCase):
    """Integration tests for VectorStore service with all backends."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create temp directories for the test
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Set environment variables for test
        os.environ["DATA_DIR"] = self.data_dir
        
        # Test documents
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
        
        # Mock the missing optional backends
        self.patch_faiss = patch("me2ai_mcp.services.backend_operations.faiss")
        self.mock_faiss = self.patch_faiss.start()
        
        self.patch_qdrant = patch("me2ai_mcp.services.vectorstore_service.QdrantClient")
        self.mock_qdrant = self.patch_qdrant.start()
        
        self.patch_pinecone = patch("me2ai_mcp.services.vectorstore_service.pinecone")
        self.mock_pinecone = self.patch_pinecone.start()
        
        # Mock the embedding functions for all models
        self.patch_sentence_transformers = patch("me2ai_mcp.services.vectorstore_service.SentenceTransformer")
        self.mock_sentence_transformers = self.patch_sentence_transformers.start()
        mock_transformer = MagicMock()
        mock_transformer.encode.return_value = np.random.rand(3, 768).astype(np.float32)
        self.mock_sentence_transformers.return_value = mock_transformer
        
        # Set up test client for each backend
        self.setup_test_clients()
    
    def tearDown(self) -> None:
        """Tear down test fixtures."""
        # Remove temp directory
        shutil.rmtree(self.temp_dir)
        
        # Stop all patches
        self.patch_faiss.stop()
        self.patch_qdrant.stop()
        self.patch_pinecone.stop()
        self.patch_sentence_transformers.stop()
    
    def setup_test_clients(self) -> None:
        """Set up test clients for each backend."""
        # ChromaDB service
        self.chroma_service = VectorStoreService(
            vector_store_type=VectorStoreType.CHROMADB,
            embedding_model=EmbeddingModel.SENTENCE_TRANSFORMERS,
            force_enable_backends=True  # For testing
        )
        self.chroma_client = TestClient(self.chroma_service.app)
        
        # FAISS service
        self.faiss_service = VectorStoreService(
            vector_store_type=VectorStoreType.FAISS,
            embedding_model=EmbeddingModel.SENTENCE_TRANSFORMERS,
            force_enable_backends=True
        )
        self.faiss_client = TestClient(self.faiss_service.app)
        
        # Qdrant service
        self.qdrant_service = VectorStoreService(
            vector_store_type=VectorStoreType.QDRANT,
            embedding_model=EmbeddingModel.SENTENCE_TRANSFORMERS,
            force_enable_backends=True
        )
        self.qdrant_client = TestClient(self.qdrant_service.app)
        
        # Pinecone service
        self.pinecone_service = VectorStoreService(
            vector_store_type=VectorStoreType.PINECONE,
            embedding_model=EmbeddingModel.SENTENCE_TRANSFORMERS,
            force_enable_backends=True
        )
        self.pinecone_client = TestClient(self.pinecone_service.app)
    
    def test_collection_management(self) -> None:
        """Test collection creation and listing across all backends."""
        backends = [
            ("ChromaDB", self.chroma_client),
            ("FAISS", self.faiss_client),
            ("Qdrant", self.qdrant_client),
            ("Pinecone", self.pinecone_client)
        ]
        
        for backend_name, client in backends:
            with self.subTest(backend=backend_name):
                # Test listing initial collections (should be empty)
                response = client.get("/collections")
                self.assertEqual(response.status_code, 200)
                initial_collections = response.json()
                
                # Create collections
                collection_names = [f"test_{backend_name.lower()}_collection_{i}" for i in range(3)]
                for name in collection_names:
                    response = client.post(f"/collections/{name}")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["message"], f"Collection '{name}' created")
                
                # Verify collections were created
                response = client.get("/collections")
                self.assertEqual(response.status_code, 200)
                new_collections = response.json()
                self.assertEqual(len(new_collections), len(collection_names))
                for name in collection_names:
                    self.assertIn(name, new_collections)
    
    def test_document_lifecycle(self) -> None:
        """Test complete document lifecycle (upsert, query, delete) across all backends."""
        backends = [
            ("ChromaDB", self.chroma_client),
            ("FAISS", self.faiss_client),
            ("Qdrant", self.qdrant_client),
            ("Pinecone", self.pinecone_client)
        ]
        
        for backend_name, client in backends:
            with self.subTest(backend=backend_name):
                collection_name = f"test_{backend_name.lower()}_lifecycle"
                
                # 1. Create collection
                response = client.post(f"/collections/{collection_name}")
                self.assertEqual(response.status_code, 200)
                
                # 2. Upsert documents
                upsert_data = {
                    "documents": self.test_docs,
                    "metadatas": self.test_metadata,
                    "ids": self.test_ids
                }
                response = client.post(f"/upsert/{collection_name}", json=upsert_data)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["count"], len(self.test_docs))
                
                # 3. Query documents (simple query)
                query_data = {
                    "query": "knowledge management",
                    "n_results": 2
                }
                response = client.post(f"/query/{collection_name}", json=query_data)
                self.assertEqual(response.status_code, 200)
                results = response.json()["results"]
                self.assertGreaterEqual(len(results), 1)
                
                # 4. Query with filters
                query_data = {
                    "query": "test",
                    "n_results": 5,
                    "where": {"category": "ai"}
                }
                response = client.post(f"/query/{collection_name}", json=query_data)
                self.assertEqual(response.status_code, 200)
                results = response.json()["results"]
                # At least one result should have category=ai
                categories = [r["metadata"]["category"] for r in results]
                self.assertIn("ai", categories)
                
                # 5. Delete by IDs
                delete_data = {
                    "ids": [self.test_ids[0]]  # Delete one document
                }
                response = client.post(f"/delete/{collection_name}", json=delete_data)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["count"], 1)
                
                # 6. Verify document was deleted
                query_data = {
                    "query": "test",
                    "n_results": 10
                }
                response = client.post(f"/query/{collection_name}", json=query_data)
                results = response.json()["results"]
                result_ids = [r["id"] for r in results]
                self.assertNotIn(self.test_ids[0], result_ids)
                
                # 7. Delete by metadata filter
                delete_data = {
                    "where": {"category": "database"}
                }
                response = client.post(f"/delete/{collection_name}", json=delete_data)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["count"], 1)
                
                # 8. Delete remaining document
                delete_data = {
                    "ids": [self.test_ids[2]]
                }
                response = client.post(f"/delete/{collection_name}", json=delete_data)
                self.assertEqual(response.status_code, 200)
                
                # 9. Verify all documents deleted
                query_data = {
                    "query": "test",
                    "n_results": 10
                }
                response = client.post(f"/query/{collection_name}", json=query_data)
                results = response.json()["results"]
                self.assertEqual(len(results), 0)
    
    def test_file_upload(self) -> None:
        """Test file upload functionality across all backends."""
        backends = [
            ("ChromaDB", self.chroma_client),
            ("FAISS", self.faiss_client),
            ("Qdrant", self.qdrant_client),
            ("Pinecone", self.pinecone_client)
        ]
        
        for backend_name, client in backends:
            with self.subTest(backend=backend_name):
                collection_name = f"test_{backend_name.lower()}_upload"
                
                # 1. Create collection
                response = client.post(f"/collections/{collection_name}")
                self.assertEqual(response.status_code, 200)
                
                # 2. Create test file content
                file_content = b"This is a test document for upload to vectorstore service.\n\nIt contains multiple paragraphs and will be embedded and stored for retrieval."
                file = {"file": ("test.txt", BytesIO(file_content), "text/plain")}
                
                # 3. Upload file
                response = client.post(
                    f"/upload/{collection_name}",
                    files=file,
                    data={"metadata": '{"source": "file_upload", "type": "text"}'}
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["count"], 1)
                document_id = response.json()["ids"][0]
                
                # 4. Query for the uploaded content
                query_data = {
                    "query": "test document upload",
                    "n_results": 1
                }
                response = client.post(f"/query/{collection_name}", json=query_data)
                self.assertEqual(response.status_code, 200)
                results = response.json()["results"]
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["id"], document_id)
                
                # 5. Delete the uploaded document
                delete_data = {
                    "ids": [document_id]
                }
                response = client.post(f"/delete/{collection_name}", json=delete_data)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["count"], 1)


# Run tests with pytest if file is executed directly
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
