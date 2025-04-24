# ME2AI MCP Vector Store Service

The ME2AI MCP Vector Store service is a scalable, pluggable microservice that provides document embedding and semantic search capabilities, supporting multiple backend vector databases.

## Features

- **Multiple Vector Store Backends**: Support for ChromaDB (default), FAISS, Qdrant, and Pinecone
- **Multiple Embedding Models**: Support for Sentence Transformers (default), OpenAI, Cohere, and HuggingFace
- **RESTful API**: Comprehensive endpoints for document management and semantic search
- **Collection Management**: Create, list, and manage vector collections
- **Document Operations**: Upsert, query, and delete documents with metadata filtering
- **File Upload**: Upload and embed document files directly
- **Comprehensive Testing**: Extensive test coverage for all backends and operations

## Backend Comparison

| Backend | Persistence | Hosted Option | Scale | Filtering | Key Strengths |
|---------|-------------|---------------|-------|-----------|---------------|
| ChromaDB | Yes (local) | No | Small-Medium | Yes | Simple setup, good for development |
| FAISS | Yes (local) | No | Medium | Limited | Fast searches, good for local deployment |
| Qdrant | Yes | Yes | Medium-Large | Advanced | Strong filtering, open-source |
| Pinecone | Yes | Yes | Large | Advanced | Production-ready, highly scalable |

## Installation

```bash
pip install me2ai-mcp
```

### Optional Dependencies

```bash
# For FAISS support
pip install faiss-cpu

# For Qdrant support
pip install qdrant-client

# For Pinecone support
pip install pinecone-client
```

## Basic Usage

### Starting the Service

```python
from me2ai_mcp.services import VectorStoreService, VectorStoreType, EmbeddingModel

# Initialize with ChromaDB backend
service = VectorStoreService(
    vector_store_type=VectorStoreType.CHROMADB,
    embedding_model=EmbeddingModel.SENTENCE_TRANSFORMERS
)

# Start the service
service.start()
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections` | GET | List all collections |
| `/collections/{name}` | POST | Create a new collection |
| `/upsert/{collection_name}` | POST | Add documents to a collection |
| `/query/{collection_name}` | POST | Query documents by similarity |
| `/delete/{collection_name}` | POST | Delete documents from a collection |
| `/upload/{collection_name}` | POST | Upload and embed a file |

## Integration with Knowledge Assistant

The Vector Store service is designed to integrate seamlessly with the ME2AI Knowledge Assistant:

### Setup

```python
from me2ai_mcp.services import VectorStoreService, VectorStoreType

# Create the service
vector_service = VectorStoreService(vector_store_type=VectorStoreType.CHROMADB)

# Start the service (Knowledge Assistant will connect to this service)
vector_service.start(host="0.0.0.0", port=8001)
```

### Knowledge Assistant Configuration

In your Knowledge Assistant application:

```python
# Configure the Knowledge Assistant to use the Vector Store service
import os

# Set the Vector Store service URL
os.environ["VECTORSTORE_URL"] = "http://localhost:8001"

# Initialize the Knowledge Assistant components
from src.components.vectorstore.store import VectorStore
vector_store = VectorStore(collection_name="knowledge_documents")
```

### Document Storage Flow

1. User uploads documents or scrapes websites in the Knowledge Assistant
2. Knowledge Assistant extracts and preprocesses text
3. Processed text is sent to the Vector Store service for embedding and storage
4. When a user asks a question, relevant documents are retrieved from the Vector Store
5. Retrieved documents are used to augment the LLM prompt for better answers

## Test Coverage

The Vector Store service has comprehensive test coverage for all backends and operations:

### Unit Tests

- **Backend Operations**: All vector store backend operations (ChromaDB, FAISS, Qdrant, Pinecone) are tested with extensive mocking and validation.
- **Service Logic**: All service endpoints, error handling, and edge cases are covered.

### Integration Tests

- **End-to-End Flow**: Complete document lifecycle across all backends.
- **Knowledge Assistant Integration**: Validation of integration patterns with the ME2AI Knowledge Assistant.

Run the test suite:

```bash
# Run all Vector Store tests with coverage reporting
python scripts/run_vectorstore_tests.py --html --all
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATA_DIR` | Directory for persistent storage | Yes |
| `OPENAI_API_KEY` | API key for OpenAI embeddings | For OpenAI model |
| `COHERE_API_KEY` | API key for Cohere embeddings | For Cohere model |
| `QDRANT_URL` | URL for Qdrant service | For Qdrant backend |
| `QDRANT_API_KEY` | API key for Qdrant service | For Qdrant backend |
| `PINECONE_API_KEY` | API key for Pinecone service | For Pinecone backend |
| `PINECONE_ENVIRONMENT` | Pinecone environment | For Pinecone backend |
| `PINECONE_INDEX` | Pinecone index name | For Pinecone backend |
