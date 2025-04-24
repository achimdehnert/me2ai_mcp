"""
Tests for backend operations of the VectorStore service.

This module tests the backend-specific operations for each vector store type
supported in the ME2AI MCP VectorStore service.
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
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

# Import the operations under test
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
    delete_pinecone,
    # Constants
    DEFAULT_DIMENSION
)


class TestHelperFunctions(unittest.TestCase):
    """Test suite for helper functions."""
    
    def test_match_metadata(self) -> None:
        """Test metadata matching function.
        
        Tests exact matching, non-matching, and edge cases.
        """
        # Test exact match
        metadata = {"source": "test", "category": "ai", "tags": ["nlp", "ml"]}
        where = {"source": "test", "category": "ai"}
        self.assertTrue(match_metadata(metadata, where))
        
        # Test non-matching key
        where_non_match_key = {"source": "test", "category": "database"}
        self.assertFalse(match_metadata(metadata, where_non_match_key))
        
        # Test non-existent key
        where_non_existent = {"source": "test", "non_existent": "value"}
        self.assertFalse(match_metadata(metadata, where_non_existent))
        
        # Test empty where condition (should match everything)
        self.assertTrue(match_metadata(metadata, {}))
        
        # Test empty metadata
        self.assertFalse(match_metadata({}, {"key": "value"}))
        
    def test_match_document(self) -> None:
        """Test document text matching function.
        
        Tests substring matching, edge cases, and multiple patterns.
        """
        # Test contains match
        text = "This is a test document about artificial intelligence and machine learning."
        where_document = {"$contains": "artificial intelligence"}
        self.assertTrue(match_document(text, where_document))
        
        # Test non-matching substring
        where_non_match = {"$contains": "blockchain"}
        self.assertFalse(match_document(text, where_non_match))
        
        # Test case sensitivity
        where_case = {"$contains": "Artificial Intelligence"}
        # Current implementation is case-sensitive
        self.assertFalse(match_document(text, where_case))
        
        # Test empty string (should match any document)
        where_empty = {"$contains": ""}
        self.assertTrue(match_document(text, where_empty))
        
        # Test unknown operator (should default to true)
        where_unknown = {"$unknown_operator": "value"}
        self.assertTrue(match_document(text, where_unknown))
        
        # Test empty document
        self.assertFalse(match_document("", {"$contains": "test"}))


class TestChromaOperations(unittest.TestCase):
    """Test suite for ChromaDB backend operations."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        # Set up mock embedding function
        self.mock_embedding_function = MagicMock()
        self.mock_embedding_function.return_value = [0.1] * DEFAULT_DIMENSION
        
        # Set up test data
        self.test_collection_name = "test_collection"
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
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_create_collection_chroma(self, mock_logger: MagicMock) -> None:
        """Test creating a collection in ChromaDB."""
        # Create mock client
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        
        # Call the function
        result = await create_collection_chroma(
            mock_client,
            self.test_collection_name,
            self.mock_embedding_function
        )
        
        # Verify results
        self.assertEqual(result, mock_collection)
        mock_client.create_collection.assert_called_once_with(
            name=self.test_collection_name,
            embedding_function=self.mock_embedding_function
        )
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_upsert_chroma_existing_collection(self, mock_logger: MagicMock) -> None:
        """Test upserting documents to an existing ChromaDB collection."""
        # Create mock client and collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        collections = {self.test_collection_name: mock_collection}
        
        # Call the function
        await upsert_chroma(
            mock_client,
            collections,
            self.test_collection_name,
            self.test_docs,
            self.test_metadata,
            self.test_ids,
            self.mock_embedding_function
        )
        
        # Verify results
        mock_collection.upsert.assert_called_once_with(
            documents=self.test_docs,
            metadatas=self.test_metadata,
            ids=self.test_ids
        )
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    @patch("me2ai_mcp.services.backend_operations.create_collection_chroma")
    async def test_upsert_chroma_new_collection(
        self, 
        mock_create: AsyncMock, 
        mock_logger: MagicMock
    ) -> None:
        """Test upserting documents to a new ChromaDB collection."""
        # Create mock client and collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_create.return_value = mock_collection
        collections = {}
        
        # Call the function
        await upsert_chroma(
            mock_client,
            collections,
            self.test_collection_name,
            self.test_docs,
            self.test_metadata,
            self.test_ids,
            self.mock_embedding_function
        )
        
        # Verify results
        mock_client.get_collection.assert_called_once_with(self.test_collection_name)
        self.assertEqual(collections[self.test_collection_name], mock_collection)
        mock_collection.upsert.assert_called_once_with(
            documents=self.test_docs,
            metadatas=self.test_metadata,
            ids=self.test_ids
        )
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    @patch("me2ai_mcp.services.backend_operations.create_collection_chroma")
    async def test_upsert_chroma_collection_not_found(
        self, 
        mock_create: AsyncMock, 
        mock_logger: MagicMock
    ) -> None:
        """Test upserting documents when collection is not found and must be created."""
        # Create mock client and collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_create.return_value = mock_collection
        collections = {}
        
        # Call the function
        await upsert_chroma(
            mock_client,
            collections,
            self.test_collection_name,
            self.test_docs,
            self.test_metadata,
            self.test_ids,
            self.mock_embedding_function
        )
        
        # Verify results
        mock_client.get_collection.assert_called_once_with(self.test_collection_name)
        mock_create.assert_called_once_with(
            mock_client,
            self.test_collection_name,
            self.mock_embedding_function
        )
        self.assertEqual(collections[self.test_collection_name], mock_collection)
        mock_collection.upsert.assert_called_once_with(
            documents=self.test_docs,
            metadatas=self.test_metadata,
            ids=self.test_ids
        )
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_query_chroma(self, mock_logger: MagicMock) -> None:
        """Test querying documents from a ChromaDB collection."""
        # Create mock collection with query results
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
            "distances": [[0.1, 0.2]]
        }
        collections = {self.test_collection_name: mock_collection}
        
        # Call the function
        results = await query_chroma(
            collections,
            self.test_collection_name,
            "test query",
            2,
            {"source": "test"},
            {"$contains": "AI"}
        )
        
        # Verify results
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=2,
            where={"source": "test"},
            where_document={"$contains": "AI"}
        )
        
        # Check result format
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "id1")
        self.assertEqual(results[0]["text"], "doc1")
        self.assertEqual(results[0]["metadata"], {"source": "test1"})
        self.assertEqual(results[0]["distance"], 0.1)
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_query_chroma_collection_not_found(self, mock_logger: MagicMock) -> None:
        """Test querying documents from a non-existent ChromaDB collection."""
        # Create empty collections dict
        collections = {}
        
        # Expect ValueError for non-existent collection
        with pytest.raises(ValueError):
            await query_chroma(
                collections,
                "nonexistent_collection",
                "test query",
                2,
                None,
                None
            )
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_delete_chroma_by_ids(self, mock_logger: MagicMock) -> None:
        """Test deleting documents by IDs from a ChromaDB collection."""
        # Create mock collection
        mock_collection = MagicMock()
        mock_collection.count.side_effect = [10, 8]  # Before and after deletion
        collections = {self.test_collection_name: mock_collection}
        
        # Call the function
        count = await delete_chroma(
            collections,
            self.test_collection_name,
            ["id1", "id2"],
            None,
            None
        )
        
        # Verify results
        mock_collection.delete.assert_called_once_with(ids=["id1", "id2"])
        self.assertEqual(count, 2)  # 10 - 8 = 2 deleted
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_delete_chroma_by_where(self, mock_logger: MagicMock) -> None:
        """Test deleting documents by metadata filter from a ChromaDB collection."""
        # Create mock collection
        mock_collection = MagicMock()
        mock_collection.count.side_effect = [10, 7]  # Before and after deletion
        collections = {self.test_collection_name: mock_collection}
        
        # Call the function
        count = await delete_chroma(
            collections,
            self.test_collection_name,
            None,
            {"source": "test"},
            None
        )
        
        # Verify results
        mock_collection.delete.assert_called_once_with(
            where={"source": "test"},
            where_document=None
        )
        self.assertEqual(count, 3)  # 10 - 7 = 3 deleted
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_delete_chroma_all(self, mock_logger: MagicMock) -> None:
        """Test deleting all documents from a ChromaDB collection."""
        # Create mock collection
        mock_collection = MagicMock()
        mock_collection.count.side_effect = [10, 0]  # Before and after deletion
        mock_collection.get.return_value = {"ids": ["id1", "id2", "id3"]}
        collections = {self.test_collection_name: mock_collection}
        
        # Call the function
        count = await delete_chroma(
            collections,
            self.test_collection_name,
            None,
            None,
            None
        )
        
        # Verify results
        mock_collection.get.assert_called_once()
        mock_collection.delete.assert_called_once_with(ids=["id1", "id2", "id3"])
        self.assertEqual(count, 10)  # 10 - 0 = 10 deleted
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_delete_chroma_collection_not_found(self, mock_logger: MagicMock) -> None:
        """Test deleting documents from a non-existent ChromaDB collection."""
        # Create empty collections dict
        collections = {}
        
        # Expect ValueError for non-existent collection
        with pytest.raises(ValueError):
            await delete_chroma(
                collections,
                "nonexistent_collection",
                None,
                None,
                None
            )


class TestFAISSOperations(unittest.TestCase):
    """Test suite for FAISS backend operations."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create temp directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Set up mock embedding function
        self.mock_embedding_function = MagicMock()
        self.mock_embedding_function.return_value = np.random.rand(3, DEFAULT_DIMENSION).astype(np.float32)
        
        # Set up test data
        self.test_collection_name = "test_collection"
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
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    @patch("me2ai_mcp.services.backend_operations.faiss")
    @patch("me2ai_mcp.services.backend_operations.pickle")
    async def test_create_collection_faiss(self, mock_pickle: MagicMock, mock_faiss: MagicMock, mock_logger: MagicMock) -> None:
        """Test creating a collection in FAISS."""
        # Set up mocks
        mock_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_index
        
        # Call the function
        result = await create_collection_faiss(
            self.test_dir,
            self.test_collection_name
        )
        
        # Verify results
        mock_faiss.IndexFlatL2.assert_called_once_with(DEFAULT_DIMENSION)
        mock_faiss.write_index.assert_called_once()
        mock_pickle.dump.assert_called_once()
        
        # Check result structure
        self.assertIn("index", result)
        self.assertIn("data", result)
        self.assertEqual(result["index"], mock_index)
        self.assertIn("ids", result["data"])
        self.assertIn("texts", result["data"])
        self.assertIn("metadatas", result["data"])
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    @patch("me2ai_mcp.services.backend_operations.faiss")
    @patch("me2ai_mcp.services.backend_operations.pickle")
    @patch("me2ai_mcp.services.backend_operations.np")
    async def test_upsert_faiss_existing_collection(self, mock_np: MagicMock, mock_pickle: MagicMock, mock_faiss: MagicMock, mock_logger: MagicMock) -> None:
        """Test upserting documents to an existing FAISS collection."""
        # Set up mocks
        mock_index = MagicMock()
        mock_np.array.return_value = np.random.rand(3, DEFAULT_DIMENSION).astype(np.float32)
        
        # Set up existing collection
        mock_data = {
            "ids": [],
            "texts": [],
            "metadatas": []
        }
        collections = {
            self.test_collection_name: {
                "index": mock_index,
                "data": mock_data
            }
        }
        
        # Call the function
        await upsert_faiss(
            self.test_dir,
            collections,
            self.test_collection_name,
            self.test_docs,
            self.test_metadata,
            self.test_ids,
            self.mock_embedding_function
        )
        
        # Verify results
        mock_np.array.assert_called_once()
        mock_index.add.assert_called_once()
        mock_faiss.write_index.assert_called_once()
        mock_pickle.dump.assert_called_once()
        
        # Verify data was updated
        collection = collections[self.test_collection_name]
        self.assertEqual(collection["data"]["ids"], self.test_ids)
        self.assertEqual(collection["data"]["texts"], self.test_docs)
        self.assertEqual(collection["data"]["metadatas"], self.test_metadata)
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    @patch("me2ai_mcp.services.backend_operations.faiss")
    @patch("me2ai_mcp.services.backend_operations.pickle")
    @patch("me2ai_mcp.services.backend_operations.np")
    @patch("me2ai_mcp.services.backend_operations.create_collection_faiss")
    async def test_upsert_faiss_new_collection(self, mock_create: AsyncMock, mock_np: MagicMock, mock_pickle: MagicMock, mock_faiss: MagicMock, mock_logger: MagicMock) -> None:
        """Test upserting documents to a new FAISS collection."""
        # Set up mocks
        mock_index = MagicMock()
        mock_np.array.return_value = np.random.rand(3, DEFAULT_DIMENSION).astype(np.float32)
        
        # Set up new collection via mock create function
        mock_data = {
            "ids": [],
            "texts": [],
            "metadatas": []
        }
        mock_create.return_value = {
            "index": mock_index,
            "data": mock_data
        }
        collections = {}
        
        # Call the function
        await upsert_faiss(
            self.test_dir,
            collections,
            self.test_collection_name,
            self.test_docs,
            self.test_metadata,
            self.test_ids,
            self.mock_embedding_function
        )
        
        # Verify create was called
        mock_create.assert_called_once_with(self.test_dir, self.test_collection_name)
        
        # Verify results
        mock_np.array.assert_called_once()
        mock_index.add.assert_called_once()
        mock_faiss.write_index.assert_called_once()
        mock_pickle.dump.assert_called_once()
        
        # Verify collection was added
        self.assertIn(self.test_collection_name, collections)
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    @patch("me2ai_mcp.services.backend_operations.np")
    async def test_query_faiss(self, mock_np: MagicMock, mock_logger: MagicMock) -> None:
        """Test querying documents from a FAISS collection."""
        # Set up mocks
        mock_index = MagicMock()
        mock_index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
        
        # Set up test data
        mock_data = {
            "ids": ["id1", "id2"],
            "texts": ["text1", "text2"],
            "metadatas": [{"source": "test1"}, {"source": "test2"}]
        }
        collections = {
            self.test_collection_name: {
                "index": mock_index,
                "data": mock_data
            }
        }
        
        # Call the function
        results = await query_faiss(
            collections,
            self.test_collection_name,
            "test query",
            2,
            None,
            None,
            self.mock_embedding_function
        )
        
        # Verify results
        mock_index.search.assert_called_once()
        
        # Check result format
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "id1")
        self.assertEqual(results[0]["text"], "text1")
        self.assertEqual(results[0]["metadata"], {"source": "test1"})
        self.assertEqual(results[0]["distance"], 0.1)
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_query_faiss_with_filters(self, mock_logger: MagicMock) -> None:
        """Test querying documents from a FAISS collection with filters."""
        # Set up mocks
        mock_index = MagicMock()
        mock_index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
        
        # Set up test data with metadata for filtering
        mock_data = {
            "ids": ["id1", "id2"],
            "texts": ["AI text", "database text"],
            "metadatas": [
                {"source": "test", "category": "ai"},
                {"source": "test", "category": "database"}
            ]
        }
        collections = {
            self.test_collection_name: {
                "index": mock_index,
                "data": mock_data
            }
        }
        
        # Call the function with metadata filter
        results = await query_faiss(
            collections,
            self.test_collection_name,
            "test query",
            2,
            {"category": "ai"},
            None,
            self.mock_embedding_function
        )
        
        # Verify results - should only return the AI document
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "id1")
        self.assertEqual(results[0]["metadata"]["category"], "ai")
        
        # Call the function with document content filter
        results = await query_faiss(
            collections,
            self.test_collection_name,
            "test query",
            2,
            None,
            {"$contains": "database"},
            self.mock_embedding_function
        )
        
        # Verify results - should only return the database document
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "id2")
        self.assertEqual(results[0]["text"], "database text")
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_query_faiss_collection_not_found(self, mock_logger: MagicMock) -> None:
        """Test querying documents from a non-existent FAISS collection."""
        # Create empty collections dict
        collections = {}
        
        # Expect ValueError for non-existent collection
        with pytest.raises(ValueError):
            await query_faiss(
                collections,
                "nonexistent_collection",
                "test query",
                2,
                None,
                None,
                self.mock_embedding_function
            )
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    @patch("me2ai_mcp.services.backend_operations.faiss")
    @patch("me2ai_mcp.services.backend_operations.pickle")
    @patch("me2ai_mcp.services.backend_operations.np")
    async def test_delete_faiss_by_ids(self, mock_np: MagicMock, mock_pickle: MagicMock, mock_faiss: MagicMock, mock_logger: MagicMock) -> None:
        """Test deleting documents by IDs from a FAISS collection."""
        # Set up mocks
        mock_index = MagicMock()
        mock_new_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_new_index
        mock_np.array.return_value = np.random.rand(1, DEFAULT_DIMENSION).astype(np.float32)
        
        # Set up test data
        mock_data = {
            "ids": ["id1", "id2", "id3"],
            "texts": ["text1", "text2", "text3"],
            "metadatas": [{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]
        }
        collections = {
            self.test_collection_name: {
                "index": mock_index,
                "data": mock_data
            }
        }
        
        # Call the function
        count = await delete_faiss(
            self.test_dir,
            collections,
            self.test_collection_name,
            ["id1"],
            None,
            None,
            self.mock_embedding_function
        )
        
        # Verify results
        mock_faiss.IndexFlatL2.assert_called_once_with(DEFAULT_DIMENSION)
        mock_np.array.assert_called_once()
        mock_new_index.add.assert_called_once()
        mock_faiss.write_index.assert_called_once()
        mock_pickle.dump.assert_called_once()
        
        # Check collection was updated
        collection = collections[self.test_collection_name]
        self.assertEqual(collection["index"], mock_new_index)
        self.assertEqual(len(collection["data"]["ids"]), 2)  # One doc deleted
        self.assertEqual(count, 1)  # One doc deleted
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    @patch("me2ai_mcp.services.backend_operations.faiss")
    @patch("me2ai_mcp.services.backend_operations.pickle")
    @patch("me2ai_mcp.services.backend_operations.np")
    async def test_delete_faiss_by_metadata(self, mock_np: MagicMock, mock_pickle: MagicMock, mock_faiss: MagicMock, mock_logger: MagicMock) -> None:
        """Test deleting documents by metadata filter from a FAISS collection."""
        # Set up mocks
        mock_index = MagicMock()
        mock_new_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_new_index
        mock_np.array.return_value = np.random.rand(2, DEFAULT_DIMENSION).astype(np.float32)
        
        # Set up test data with metadata for filtering
        mock_data = {
            "ids": ["id1", "id2", "id3"],
            "texts": ["text1", "text2", "text3"],
            "metadatas": [
                {"source": "test", "category": "ai"},
                {"source": "test", "category": "database"},
                {"source": "test", "category": "ai"}
            ]
        }
        collections = {
            self.test_collection_name: {
                "index": mock_index,
                "data": mock_data
            }
        }
        
        # Call the function with metadata filter to delete AI documents
        count = await delete_faiss(
            self.test_dir,
            collections,
            self.test_collection_name,
            None,
            {"category": "ai"},
            None,
            self.mock_embedding_function
        )
        
        # Verify results
        mock_faiss.IndexFlatL2.assert_called_once_with(DEFAULT_DIMENSION)
        # Should rebuild index with only non-AI documents
        mock_np.array.assert_called_once()
        mock_new_index.add.assert_called_once()
        mock_faiss.write_index.assert_called_once()
        mock_pickle.dump.assert_called_once()
        
        # Check collection was updated
        collection = collections[self.test_collection_name]
        self.assertEqual(collection["index"], mock_new_index)
        # Should have 2 AI docs removed
        self.assertEqual(len(collection["data"]["ids"]), 1)
        self.assertEqual(count, 2)
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_delete_faiss_collection_not_found(self, mock_logger: MagicMock) -> None:
        """Test deleting documents from a non-existent FAISS collection."""
        # Create empty collections dict
        collections = {}
        
        # Expect ValueError for non-existent collection
        with pytest.raises(ValueError):
            await delete_faiss(
                self.test_dir,
                collections,
                "nonexistent_collection",
                None,
                None,
                None,
                self.mock_embedding_function
            )


class TestQdrantOperations(unittest.TestCase):
    """Test suite for Qdrant backend operations."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        # Set up mock embedding function
        self.mock_embedding_function = MagicMock()
        self.mock_embedding_function.return_value = np.random.rand(3, DEFAULT_DIMENSION).astype(np.float32)
        
        # Set up test data
        self.test_collection_name = "test_collection"
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
    
    @patch("me2ai_mcp.services.backend_operations.models")
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_create_collection_qdrant(self, mock_logger: MagicMock, mock_models: MagicMock) -> None:
        """Test creating a collection in Qdrant."""
        # Create mock client
        mock_client = MagicMock()
        mock_vector_params = MagicMock()
        mock_models.VectorParams.return_value = mock_vector_params
        
        # Call the function
        await create_collection_qdrant(
            mock_client,
            self.test_collection_name
        )
        
        # Verify results
        mock_models.VectorParams.assert_called_once_with(
            size=DEFAULT_DIMENSION,
            distance=DEFAULT_DISTANCE
        )
        mock_client.create_collection.assert_called_once_with(
            collection_name=self.test_collection_name,
            vectors_config=mock_vector_params
        )
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.models")
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_upsert_qdrant(self, mock_logger: MagicMock, mock_models: MagicMock) -> None:
        """Test upserting documents to a Qdrant collection."""
        # Create mock client and point struct
        mock_client = MagicMock()
        mock_point = MagicMock()
        mock_models.PointStruct.side_effect = lambda id, vector, payload: mock_point
        
        # Call the function
        await upsert_qdrant(
            mock_client,
            self.test_collection_name,
            self.test_docs,
            self.test_metadata,
            self.test_ids,
            self.mock_embedding_function
        )
        
        # Verify results
        self.mock_embedding_function.assert_called_once_with(self.test_docs)
        mock_models.PointStruct.assert_called()
        mock_client.upsert.assert_called_once_with(
            collection_name=self.test_collection_name,
            points=[mock_point, mock_point, mock_point]
        )
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.models")
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_query_qdrant_no_filter(self, mock_logger: MagicMock, mock_models: MagicMock) -> None:
        """Test querying documents from a Qdrant collection without filters."""
        # Create mock client and search results
        mock_client = MagicMock()
        
        # Create mock search results
        mock_result1 = MagicMock()
        mock_result1.id = self.test_ids[0]
        mock_result1.payload = {"text": self.test_docs[0], "source": "test", "category": "ai"}
        mock_result1.score = 0.95
        
        mock_result2 = MagicMock()
        mock_result2.id = self.test_ids[1]
        mock_result2.payload = {"text": self.test_docs[1], "source": "test", "category": "database"}
        mock_result2.score = 0.85
        
        mock_client.search.return_value = [mock_result1, mock_result2]
        
        # Call the function
        results = await query_qdrant(
            mock_client,
            self.test_collection_name,
            "test query",
            2,
            None,
            None,
            self.mock_embedding_function
        )
        
        # Verify results
        self.mock_embedding_function.assert_called_once_with("test query")
        mock_client.search.assert_called_once()
        
        # Check result format
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], self.test_ids[0])
        self.assertEqual(results[0]["text"], self.test_docs[0])
        self.assertEqual(results[0]["metadata"]["category"], "ai")
        self.assertEqual(results[0]["distance"], 0.95)
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.models")
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_query_qdrant_with_filter(self, mock_logger: MagicMock, mock_models: MagicMock) -> None:
        """Test querying documents from a Qdrant collection with metadata filter."""
        # Create mock client, filter, and search results
        mock_client = MagicMock()
        mock_filter = MagicMock()
        mock_field_condition = MagicMock()
        mock_match_value = MagicMock()
        
        mock_models.FieldCondition.return_value = mock_field_condition
        mock_models.MatchValue.return_value = mock_match_value
        mock_models.Filter.return_value = mock_filter
        
        # Create mock search results
        mock_result = MagicMock()
        mock_result.id = self.test_ids[0]
        mock_result.payload = {"text": self.test_docs[0], "source": "test", "category": "ai"}
        mock_result.score = 0.95
        
        mock_client.search.return_value = [mock_result]
        
        # Call the function with metadata filter
        results = await query_qdrant(
            mock_client,
            self.test_collection_name,
            "test query",
            2,
            {"category": "ai"},
            None,
            self.mock_embedding_function
        )
        
        # Verify results
        self.mock_embedding_function.assert_called_once_with("test query")
        mock_models.FieldCondition.assert_called_once_with(
            key="category",
            match=mock_match_value
        )
        mock_models.MatchValue.assert_called_once_with(value="ai")
        mock_models.Filter.assert_called_once_with(must=[mock_field_condition])
        mock_client.search.assert_called_once_with(
            collection_name=self.test_collection_name,
            query_vector=self.mock_embedding_function.return_value,
            limit=2,
            filter=mock_filter
        )
        
        # Check result format
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], self.test_ids[0])
        self.assertEqual(results[0]["text"], self.test_docs[0])
        self.assertEqual(results[0]["metadata"]["category"], "ai")
        self.assertEqual(results[0]["distance"], 0.95)
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.models")
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_query_qdrant_with_document_filter(self, mock_logger: MagicMock, mock_models: MagicMock) -> None:
        """Test querying documents from a Qdrant collection with document content filter."""
        # Create mock client and search results
        mock_client = MagicMock()
        
        # Create mock search results
        mock_result1 = MagicMock()
        mock_result1.id = self.test_ids[0]
        mock_result1.payload = {"text": "This is about artificial intelligence", "source": "test", "category": "ai"}
        mock_result1.score = 0.95
        
        mock_result2 = MagicMock()
        mock_result2.id = self.test_ids[1]
        mock_result2.payload = {"text": "This is about database technology", "source": "test", "category": "database"}
        mock_result2.score = 0.85
        
        mock_client.search.return_value = [mock_result1, mock_result2]
        
        # Call the function with document content filter
        results = await query_qdrant(
            mock_client,
            self.test_collection_name,
            "test query",
            2,
            None,
            {"$contains": "database"},
            self.mock_embedding_function
        )
        
        # Verify results
        self.mock_embedding_function.assert_called_once_with("test query")
        mock_client.search.assert_called_once()
        
        # Check result format - should only include the database document
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], self.test_ids[1])
        self.assertEqual(results[0]["text"], "This is about database technology")
        self.assertEqual(results[0]["metadata"]["category"], "database")
        self.assertEqual(results[0]["distance"], 0.85)
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.models")
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_delete_qdrant_by_ids(self, mock_logger: MagicMock, mock_models: MagicMock) -> None:
        """Test deleting documents by IDs from a Qdrant collection."""
        # Create mock client and point ids list
        mock_client = MagicMock()
        mock_points_selector = MagicMock()
        mock_models.PointIdsList.return_value = mock_points_selector
        
        # Mock count results
        mock_count_before = MagicMock(count=10)
        mock_count_after = MagicMock(count=8)
        mock_client.count.side_effect = [mock_count_before, mock_count_after]
        
        # Call the function
        count = await delete_qdrant(
            mock_client,
            self.test_collection_name,
            ["id1", "id2"],
            None,
            None
        )
        
        # Verify results
        mock_models.PointIdsList.assert_called_once_with(points=["id1", "id2"])
        mock_client.delete.assert_called_once_with(
            collection_name=self.test_collection_name,
            points_selector=mock_points_selector
        )
        self.assertEqual(count, 2)  # 10 - 8 = 2 deleted
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.models")
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_delete_qdrant_by_filter(self, mock_logger: MagicMock, mock_models: MagicMock) -> None:
        """Test deleting documents by metadata filter from a Qdrant collection."""
        # Create mock client, filter, and filter selector
        mock_client = MagicMock()
        mock_filter = MagicMock()
        mock_filter_selector = MagicMock()
        mock_field_condition = MagicMock()
        mock_match_value = MagicMock()
        
        mock_models.FieldCondition.return_value = mock_field_condition
        mock_models.MatchValue.return_value = mock_match_value
        mock_models.Filter.return_value = mock_filter
        mock_models.FilterSelector.return_value = mock_filter_selector
        
        # Mock count results
        mock_count_before = MagicMock(count=10)
        mock_count_after = MagicMock(count=7)
        mock_client.count.side_effect = [mock_count_before, mock_count_after]
        
        # Call the function with metadata filter
        count = await delete_qdrant(
            mock_client,
            self.test_collection_name,
            None,
            {"category": "ai"},
            None
        )
        
        # Verify results
        mock_models.FieldCondition.assert_called_once_with(
            key="category",
            match=mock_match_value
        )
        mock_models.MatchValue.assert_called_once_with(value="ai")
        mock_models.Filter.assert_called_once_with(must=[mock_field_condition])
        mock_models.FilterSelector.assert_called_once_with(filter=mock_filter)
        mock_client.delete.assert_called_once_with(
            collection_name=self.test_collection_name,
            points_selector=mock_filter_selector
        )
        self.assertEqual(count, 3)  # 10 - 7 = 3 deleted
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.models")
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_delete_qdrant_by_document_filter(self, mock_logger: MagicMock, mock_models: MagicMock) -> None:
        """Test deleting documents by document content filter from a Qdrant collection."""
        # Create mock client and scroll results
        mock_client = MagicMock()
        
        # Create mock point results
        mock_point1 = MagicMock()
        mock_point1.id = "id1"
        mock_point1.payload = {"text": "This is about artificial intelligence", "category": "ai"}
        
        mock_point2 = MagicMock()
        mock_point2.id = "id2"
        mock_point2.payload = {"text": "This is about database technology", "category": "database"}
        
        mock_client.scroll.return_value = [[mock_point1, mock_point2]]
        
        # Mock count results
        mock_count_before = MagicMock(count=10)
        mock_count_after = MagicMock(count=9)
        mock_client.count.side_effect = [mock_count_before, mock_count_after]
        
        # Create mock points selector for deletion
        mock_points_selector = MagicMock()
        mock_models.PointIdsList.return_value = mock_points_selector
        
        # Call the function with document content filter
        count = await delete_qdrant(
            mock_client,
            self.test_collection_name,
            None,
            None,
            {"$contains": "intelligence"}
        )
        
        # Verify results
        mock_client.scroll.assert_called_once_with(
            collection_name=self.test_collection_name,
            limit=1000,
            with_payload=True
        )
        mock_models.PointIdsList.assert_called_once_with(points=["id1"])
        mock_client.delete.assert_called_once_with(
            collection_name=self.test_collection_name,
            points_selector=mock_points_selector
        )
        self.assertEqual(count, 1)  # 10 - 9 = 1 deleted
        mock_logger.info.assert_called_once()


class TestPineconeOperations(unittest.TestCase):
    """Test suite for Pinecone backend operations."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        # Set up mock embedding function
        self.mock_embedding_function = MagicMock()
        self.mock_embedding_function.return_value = np.random.rand(3, DEFAULT_DIMENSION).astype(np.float32)
        
        # Set up test data
        self.test_collection_name = "test_namespace"  # For Pinecone, collections are namespaces
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
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_upsert_pinecone(self, mock_logger: MagicMock) -> None:
        """Test upserting documents to a Pinecone index."""
        # Create mock index
        mock_index = MagicMock()
        
        # Call the function
        await upsert_pinecone(
            mock_index,
            self.test_collection_name,
            self.test_docs,
            self.test_metadata,
            self.test_ids,
            self.mock_embedding_function
        )
        
        # Verify results
        self.mock_embedding_function.assert_called_once_with(self.test_docs)
        
        # Verify mock_index.upsert was called with the right parameters
        # Since we use batch processing, check if upsert was called
        mock_index.upsert.assert_called_once()
        call_args = mock_index.upsert.call_args
        call_kwargs = call_args[1]
        
        # Check namespace parameter
        self.assertEqual(call_kwargs["namespace"], self.test_collection_name)
        
        # Verify vectors parameter contains the right structure
        vectors = call_kwargs["vectors"]
        self.assertEqual(len(vectors), 3)  # 3 documents
        
        # Check a vector entry structure
        vector_entry = vectors[0]
        self.assertIn("id", vector_entry)
        self.assertIn("values", vector_entry)
        self.assertIn("metadata", vector_entry)
        
        # Verify metadata includes the document text
        self.assertIn("text", vector_entry["metadata"])
        
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_query_pinecone_no_filter(self, mock_logger: MagicMock) -> None:
        """Test querying documents from a Pinecone index without filters."""
        # Create mock index and query results
        mock_index = MagicMock()
        
        # Create mock query results
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": self.test_ids[0],
                    "score": 0.95,
                    "metadata": {
                        "text": self.test_docs[0],
                        "source": "test",
                        "category": "ai"
                    }
                },
                {
                    "id": self.test_ids[1],
                    "score": 0.85,
                    "metadata": {
                        "text": self.test_docs[1],
                        "source": "test",
                        "category": "database"
                    }
                }
            ]
        }
        
        # Call the function
        results = await query_pinecone(
            mock_index,
            self.test_collection_name,
            "test query",
            2,
            None,
            None,
            self.mock_embedding_function
        )
        
        # Verify results
        self.mock_embedding_function.assert_called_once_with("test query")
        mock_index.query.assert_called_once_with(
            vector=self.mock_embedding_function.return_value,
            namespace=self.test_collection_name,
            top_k=2,
            include_metadata=True,
            filter=None
        )
        
        # Check result format
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], self.test_ids[0])
        self.assertEqual(results[0]["text"], self.test_docs[0])
        self.assertEqual(results[0]["metadata"]["category"], "ai")
        self.assertEqual(results[0]["distance"], 0.95)
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_query_pinecone_with_filter(self, mock_logger: MagicMock) -> None:
        """Test querying documents from a Pinecone index with metadata filter."""
        # Create mock index and query results
        mock_index = MagicMock()
        
        # Create mock query results
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": self.test_ids[0],
                    "score": 0.95,
                    "metadata": {
                        "text": self.test_docs[0],
                        "source": "test",
                        "category": "ai"
                    }
                }
            ]
        }
        
        # Call the function with metadata filter
        results = await query_pinecone(
            mock_index,
            self.test_collection_name,
            "test query",
            2,
            {"category": "ai"},
            None,
            self.mock_embedding_function
        )
        
        # Verify results
        self.mock_embedding_function.assert_called_once_with("test query")
        mock_index.query.assert_called_once_with(
            vector=self.mock_embedding_function.return_value,
            namespace=self.test_collection_name,
            top_k=2,
            include_metadata=True,
            filter={"category": "ai"}
        )
        
        # Check result format
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], self.test_ids[0])
        self.assertEqual(results[0]["text"], self.test_docs[0])
        self.assertEqual(results[0]["metadata"]["category"], "ai")
        self.assertEqual(results[0]["distance"], 0.95)
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_query_pinecone_with_document_filter(self, mock_logger: MagicMock) -> None:
        """Test querying documents from a Pinecone index with document content filter."""
        # Create mock index and query results
        mock_index = MagicMock()
        
        # Create mock query results
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": self.test_ids[0],
                    "score": 0.95,
                    "metadata": {
                        "text": "This is about artificial intelligence",
                        "source": "test",
                        "category": "ai"
                    }
                },
                {
                    "id": self.test_ids[1],
                    "score": 0.85,
                    "metadata": {
                        "text": "This is about database technology",
                        "source": "test",
                        "category": "database"
                    }
                }
            ]
        }
        
        # Call the function with document content filter
        results = await query_pinecone(
            mock_index,
            self.test_collection_name,
            "test query",
            2,
            None,
            {"$contains": "database"},
            self.mock_embedding_function
        )
        
        # Verify results
        self.mock_embedding_function.assert_called_once_with("test query")
        mock_index.query.assert_called_once()
        
        # Check result format - should only include the database document
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], self.test_ids[1])
        self.assertEqual(results[0]["text"], "This is about database technology")
        self.assertEqual(results[0]["metadata"]["category"], "database")
        self.assertEqual(results[0]["distance"], 0.85)
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_delete_pinecone_by_ids(self, mock_logger: MagicMock) -> None:
        """Test deleting documents by IDs from a Pinecone index."""
        # Create mock index
        mock_index = MagicMock()
        
        # Mock describe_index_stats results
        mock_index.describe_index_stats.side_effect = [
            # Before deletion
            {"namespaces": {self.test_collection_name: {"vector_count": 10}}},
            # After deletion
            {"namespaces": {self.test_collection_name: {"vector_count": 8}}}
        ]
        
        # Call the function
        count = await delete_pinecone(
            mock_index,
            self.test_collection_name,
            ["id1", "id2"],
            None,
            None
        )
        
        # Verify results
        mock_index.delete.assert_called_once_with(
            ids=["id1", "id2"],
            namespace=self.test_collection_name
        )
        mock_index.describe_index_stats.assert_called()
        self.assertEqual(count, 2)  # 10 - 8 = 2 deleted
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_delete_pinecone_by_filter(self, mock_logger: MagicMock) -> None:
        """Test deleting documents by metadata filter from a Pinecone index."""
        # Create mock index
        mock_index = MagicMock()
        
        # Mock query results for matching filter
        mock_index.query.return_value = {
            "matches": [
                {"id": "id1", "score": 0.9},
                {"id": "id3", "score": 0.8}
            ]
        }
        
        # Mock describe_index_stats results
        mock_index.describe_index_stats.side_effect = [
            # Before deletion
            {"namespaces": {self.test_collection_name: {"vector_count": 10}}},
            # After deletion
            {"namespaces": {self.test_collection_name: {"vector_count": 8}}}
        ]
        
        # Call the function with metadata filter
        count = await delete_pinecone(
            mock_index,
            self.test_collection_name,
            None,
            {"category": "ai"},
            None
        )
        
        # Verify results
        mock_index.query.assert_called_once()
        query_args = mock_index.query.call_args[1]
        self.assertEqual(query_args["namespace"], self.test_collection_name)
        self.assertEqual(query_args["filter"], {"category": "ai"})
        
        # Check delete called with the IDs from the query result
        mock_index.delete.assert_called_once_with(
            ids=["id1", "id3"],
            namespace=self.test_collection_name
        )
        self.assertEqual(count, 2)  # 10 - 8 = 2 deleted
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_delete_pinecone_by_document_filter(self, mock_logger: MagicMock) -> None:
        """Test deleting documents by document content filter from a Pinecone index."""
        # Create mock index
        mock_index = MagicMock()
        
        # Mock query results for all documents
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": "id1",
                    "score": 0.9,
                    "metadata": {"text": "This is about artificial intelligence"}
                },
                {
                    "id": "id2",
                    "score": 0.8,
                    "metadata": {"text": "This is about database technology"}
                }
            ]
        }
        
        # Mock describe_index_stats results
        mock_index.describe_index_stats.side_effect = [
            # Before deletion
            {"namespaces": {self.test_collection_name: {"vector_count": 10}}},
            # After deletion
            {"namespaces": {self.test_collection_name: {"vector_count": 9}}}
        ]
        
        # Call the function with document content filter
        count = await delete_pinecone(
            mock_index,
            self.test_collection_name,
            None,
            None,
            {"$contains": "intelligence"}
        )
        
        # Verify results
        mock_index.query.assert_called_once()
        
        # Check delete called with the IDs matching the document filter
        mock_index.delete.assert_called_once_with(
            ids=["id1"],  # Only id1 contains "intelligence"
            namespace=self.test_collection_name
        )
        self.assertEqual(count, 1)  # 10 - 9 = 1 deleted
        mock_logger.info.assert_called_once()
    
    @patch("me2ai_mcp.services.backend_operations.logger")
    async def test_delete_pinecone_empty_namespace(self, mock_logger: MagicMock) -> None:
        """Test deleting from a namespace that doesn't exist in Pinecone."""
        # Create mock index
        mock_index = MagicMock()
        
        # Mock describe_index_stats for empty or no namespace
        mock_index.describe_index_stats.side_effect = [
            # No namespaces
            {"namespaces": {}},
            # Still no namespaces
            {"namespaces": {}}
        ]
        
        # Call the function
        count = await delete_pinecone(
            mock_index,
            "nonexistent_namespace",
            ["id1"],
            None,
            None
        )
        
        # Verify results
        mock_index.delete.assert_called_once()
        self.assertEqual(count, 0)  # No vectors deleted
        mock_logger.info.assert_called_once()


# Run tests with pytest
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
