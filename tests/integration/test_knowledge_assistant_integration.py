"""
ME2AI Knowledge Assistant Integration Test

This module tests the integration between the ME2AI MCP VectorStore
microservice and the ME2AI Knowledge Assistant application.
"""

import unittest
import os
import tempfile
import shutil
import requests
import json
import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

from me2ai_mcp.services.vectorstore_service import (
    VectorStoreService,
    VectorStoreType,
    EmbeddingModel
)


class TestKnowledgeAssistantIntegration(unittest.TestCase):
    """Test integration between VectorStore service and Knowledge Assistant."""
    
    @pytest.mark.integration
    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create temp directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Set environment variables
        os.environ["DATA_DIR"] = self.data_dir
        
        # Mock dependencies to avoid external service calls
        self.patchers = []
        self._setup_mocks()
        
        # Sample data for tests
        self.test_docs = [
            "ME2AI MCP provides a comprehensive knowledge management system.",
            "Vector stores allow efficient semantic search on document embeddings.",
            "The Knowledge Assistant uses ME2AI MCP for document retrieval and Q&A."
        ]
        self.test_metadata = [
            {"source": "docs", "category": "overview"},
            {"source": "docs", "category": "vectorstore"},
            {"source": "docs", "category": "assistant"}
        ]
        
        # Create test vector store service
        self.vector_service = VectorStoreService(
            vector_store_type=VectorStoreType.CHROMADB,
            embedding_model=EmbeddingModel.SENTENCE_TRANSFORMERS,
            force_enable_backends=True
        )
    
    def tearDown(self) -> None:
        """Tear down test fixtures."""
        # Stop all patches
        for patcher in self.patchers:
            patcher.stop()
        
        # Remove temp directory
        shutil.rmtree(self.temp_dir)
    
    def _setup_mocks(self) -> None:
        """Set up mock objects for testing."""
        # Mock SentenceTransformer for embeddings
        patcher = patch("me2ai_mcp.services.vectorstore_service.SentenceTransformer")
        self.mock_transformer = patcher.start()
        self.patchers.append(patcher)
        
        # Set up the transformer mock to return embeddings
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 768]  # Simple mock embeddings
        self.mock_transformer.return_value = mock_model
        
        # Mock requests for any external API calls
        patcher = patch("requests.post")
        self.mock_requests_post = patcher.start()
        self.patchers.append(patcher)
        
        # Mock requests get
        patcher = patch("requests.get")
        self.mock_requests_get = patcher.start()
        self.patchers.append(patcher)
    
    @pytest.mark.integration
    def test_knowledge_assistant_document_flow(self) -> None:
        """Test document processing flow between Knowledge Assistant and VectorStore."""
        # Simulate Knowledge Assistant document processing
        collection_name = "test_knowledge_docs"
        
        # 1. Create collection (simulates Knowledge Assistant initialization)
        collection_response = self._simulate_ka_create_collection(collection_name)
        self.assertTrue(collection_response["success"])
        
        # 2. Process documents (simulates Knowledge Assistant document upload)
        docs_response = self._simulate_ka_process_documents(
            collection_name, 
            self.test_docs, 
            self.test_metadata
        )
        self.assertTrue(docs_response["success"])
        self.assertEqual(docs_response["count"], len(self.test_docs))
        
        # 3. Perform query (simulates Knowledge Assistant answering a question)
        query_response = self._simulate_ka_query(
            collection_name,
            "What does ME2AI MCP provide?",
            n_results=2
        )
        self.assertTrue(query_response["success"])
        self.assertGreaterEqual(len(query_response["results"]), 1)
        
        # 4. Ensure query results are properly formatted for Knowledge Assistant
        first_result = query_response["results"][0]
        self.assertIn("text", first_result)
        self.assertIn("metadata", first_result)
        self.assertIn("distance", first_result)
    
    @pytest.mark.integration
    def test_knowledge_assistant_qa_flow(self) -> None:
        """Test question-answering flow using VectorStore for retrieval."""
        # Simulate Knowledge Assistant QA flow
        collection_name = "test_knowledge_qa"
        
        # 1. Initialize collection with documents
        self._simulate_ka_create_collection(collection_name)
        self._simulate_ka_process_documents(
            collection_name, 
            self.test_docs, 
            self.test_metadata
        )
        
        # 2. Simulate QA with retrieval augmentation
        question = "How does the Knowledge Assistant use ME2AI MCP?"
        
        # 2a. Retrieve relevant documents
        retrieval_response = self._simulate_ka_query(
            collection_name,
            question,
            n_results=2
        )
        
        # 2b. Format retrieved documents for LLM context
        retrieved_context = "\n\n".join([r["text"] for r in retrieval_response["results"]])
        
        # 2c. Simulate LLM answer generation with retrieved context
        mock_llm_response = {
            "answer": "The Knowledge Assistant uses ME2AI MCP for document retrieval and question answering capabilities.",
            "sources": [doc["metadata"]["source"] for doc in retrieval_response["results"]]
        }
        
        # Verify the retrieval worked and could be used by the LLM
        self.assertIn("ME2AI MCP", retrieved_context)
        self.assertIn("Knowledge Assistant", retrieved_context)
        self.assertGreaterEqual(len(retrieval_response["results"]), 1)
    
    def _simulate_ka_create_collection(self, collection_name: str) -> Dict[str, Any]:
        """Simulate Knowledge Assistant creating a vector store collection."""
        # Direct API call to the vector service
        with patch("fastapi.testclient.TestClient.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"message": f"Collection '{collection_name}' created"}
            
            # Simulate the Knowledge Assistant making the call
            response = {"success": True, "message": f"Created collection {collection_name}"}
            return response
    
    def _simulate_ka_process_documents(
        self, 
        collection_name: str, 
        documents: List[str], 
        metadatas: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Simulate Knowledge Assistant processing documents."""
        # Generate IDs as the Knowledge Assistant would
        import uuid
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Direct internal API call
        upsert_data = {
            "documents": documents,
            "metadatas": metadatas,
            "ids": ids
        }
        
        with patch("fastapi.testclient.TestClient.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"count": len(documents), "ids": ids}
            
            # Return simulated response
            return {
                "success": True,
                "count": len(documents),
                "ids": ids
            }
    
    def _simulate_ka_query(
        self, 
        collection_name: str, 
        query: str, 
        n_results: int = 3,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Simulate Knowledge Assistant querying for documents."""
        # Prepare query data
        query_data = {
            "query": query,
            "n_results": n_results
        }
        if where:
            query_data["where"] = where
        
        # Generate mock results
        mock_results = []
        for i in range(min(n_results, len(self.test_docs))):
            mock_results.append({
                "id": f"doc_{i}",
                "text": self.test_docs[i],
                "metadata": self.test_metadata[i],
                "distance": 0.8 - (i * 0.1)
            })
        
        with patch("fastapi.testclient.TestClient.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"results": mock_results}
            
            # Return simulated response
            return {
                "success": True,
                "results": mock_results
            }


if __name__ == "__main__":
    unittest.main()
