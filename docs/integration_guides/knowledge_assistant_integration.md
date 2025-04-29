# ME2AI Knowledge Assistant Integration Guide

This guide explains how to effectively integrate ME2AI MCP components with the ME2AI Knowledge Assistant application.

## Overview

The ME2AI Knowledge Assistant leverages several key components from the ME2AI MCP package:

1. Vector Store Management (ChromaDB)
2. Embedding Services
3. OpenAI Client Integration
4. Document Processing
5. Web Scraping

The modularized architecture of ME2AI MCP makes it easier to use these components in the Knowledge Assistant application.

## Setup

### Installation

First, ensure you have the latest version of ME2AI MCP:

```bash
pip install me2ai-mcp[all]
```

For a minimal installation focused on Knowledge Assistant needs:

```bash
pip install me2ai-mcp[embedding,vectorstore,llm,web]
```

### Environment Configuration

Create a `.env` file with necessary API keys:

```text
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key  # Optional
```

## Vector Store Integration

The Knowledge Assistant primarily uses ChromaDB as its vector store. The modularized handlers make this integration cleaner:

```python
from me2ai_mcp.services.handlers import upsert_chroma, query_chroma, delete_chroma

# Create a collection
collection_name = "knowledge_assistant_docs"
embedding_dimension = 1536  # For OpenAI embeddings

# Store documents
documents = ["Document content 1", "Document content 2"]
metadatas = [{"source": "pdf", "title": "Doc 1"}, {"source": "web", "title": "Doc 2"}]
embeddings = [...] # Your embeddings here
await upsert_chroma(
    collection_name=collection_name, 
    texts=documents, 
    metadatas=metadatas,
    embeddings=embeddings
)

# Query documents
query_embedding = [...]  # Your query embedding
results = await query_chroma(
    collection_name=collection_name,
    query_embeddings=[query_embedding],
    n_results=5
)
```

## Embedding Service

The Knowledge Assistant needs to generate embeddings for documents and queries. Use the modularized embedding service:

```python
from me2ai_mcp.adapters.embedding.strategy import create_embedding_strategy

# Create an embedding strategy (OpenAI is used in the Knowledge Assistant)
embedding_strategy = create_embedding_strategy(
    provider="openai",
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Embed a document
document_text = "This is a document about machine learning."
embedding = await embedding_strategy.embed_text([document_text])

# Batch embed multiple documents
documents = ["Doc 1", "Doc 2", "Doc 3"]
embeddings = await embedding_strategy.embed_text(documents)
```

## OpenAI Client Integration

The Knowledge Assistant uses the OpenAI client for question answering:

```python
from me2ai_mcp.clients.openai_client import OpenAIMCPClient

# Initialize client
client = OpenAIMCPClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o"
)

# Connect to the API
await client.connect()

# Get list of available tools
tools = await client.list_tools()

# Create a prompt with context from vector store
documents = [...]  # Retrieved from vector store
prompt = f"""
Answer the following question based on these documents:
{documents}

Question: {user_question}
"""

# Get completion
response = await client.get_completion(prompt)

# Disconnect
await client.disconnect()
```

## Web Scraping Integration

The Knowledge Assistant uses web scraping tools to extract content from websites:

```python
from me2ai_mcp.tools.web_content_tool import WebContentTool

# Initialize the web content tool
web_tool = WebContentTool()

# Extract content from a URL
url = "https://example.com/page"
content = await web_tool.extract_content(url)

# Handle the extracted content
if content.get("status") == "success":
    text = content.get("text", "")
    title = content.get("title", "")
    # Process the content for the Knowledge Assistant
else:
    # Handle error
```

## Testing Integration

To ensure compatibility with the Knowledge Assistant, run the specific tests:

```bash
# Activate the robot environment
source .venv-robot/bin/activate  # On Windows: .venv-robot\Scripts\activate

# Run Knowledge Assistant compatibility tests
robot --outputdir results --include knowledge-assistant tests/robot/
```

## Best Practices

1. **Error Handling**: Implement comprehensive error handling, particularly for API-dependent services like OpenAI and embedding services.

2. **Caching**: Consider implementing caching for embeddings and vector store results to improve performance.

3. **Asynchronous Operations**: Use the asynchronous interfaces provided by ME2AI MCP for improved performance.

4. **Security**: Never hardcode API keys; always use environment variables.

5. **Provider Abstraction**: Use the factory functions and protocols to make your code provider-agnostic.

## Example: Complete Knowledge Assistant Integration

```python
import os
from dotenv import load_dotenv
from me2ai_mcp.adapters.embedding.strategy import create_embedding_strategy
from me2ai_mcp.services.handlers import upsert_chroma, query_chroma
from me2ai_mcp.clients.openai_client import OpenAIMCPClient
from me2ai_mcp.tools.web_content_tool import WebContentTool

# Load environment variables
load_dotenv()

async def process_document(text, metadata):
    """Process a document for the Knowledge Assistant."""
    # Create embedding strategy
    embedding_strategy = create_embedding_strategy(
        provider="openai",
        model="text-embedding-3-small"
    )
    
    # Generate embedding
    embeddings = await embedding_strategy.embed_text([text])
    
    # Store in vector database
    await upsert_chroma(
        collection_name="knowledge_assistant",
        texts=[text],
        metadatas=[metadata],
        embeddings=embeddings
    )
    
    return True

async def answer_question(question):
    """Answer a question using the Knowledge Assistant."""
    # Create embedding for the question
    embedding_strategy = create_embedding_strategy(
        provider="openai",
        model="text-embedding-3-small"
    )
    
    question_embedding = await embedding_strategy.embed_text([question])
    
    # Query vector store
    results = await query_chroma(
        collection_name="knowledge_assistant",
        query_embeddings=question_embedding,
        n_results=5
    )
    
    # Extract relevant documents
    documents = [item["text"] for item in results[0]["documents"]]
    
    # Initialize OpenAI client
    client = OpenAIMCPClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o"
    )
    
    # Connect to API
    await client.connect()
    
    # Create prompt with context
    context = "\n\n".join(documents)
    prompt = f"""
    Answer the following question based on these documents:
    
    {context}
    
    Question: {question}
    """
    
    # Get completion
    response = await client.get_completion(prompt)
    
    # Disconnect
    await client.disconnect()
    
    return response["choices"][0]["message"]["content"]

async def scrape_website(url):
    """Scrape content from a website."""
    web_tool = WebContentTool()
    content = await web_tool.extract_content(url)
    
    if content.get("status") == "success":
        # Process the scraped content
        text = content.get("text", "")
        title = content.get("title", "")
        
        # Store in Knowledge Assistant
        await process_document(
            text=text,
            metadata={"source": "web", "url": url, "title": title}
        )
        
        return True
    else:
        return False
```

## Troubleshooting

### Common Issues

1. **Vector Store Connection Errors**: Ensure ChromaDB is properly installed and initialized.
2. **Embedding Dimension Mismatch**: Verify that your ChromaDB collection's dimension matches your embedding model's output dimension.
3. **API Key Issues**: Check that environment variables are properly loaded.

### Logging

Enable debug logging for ME2AI MCP components:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Upgrading

When upgrading ME2AI MCP, check the release notes for any breaking changes, particularly in the following areas:

1. Vector store handler interfaces
2. Embedding strategy protocols
3. OpenAI client parameters

The modular architecture should minimize breaking changes, but always test thoroughly after upgrading.
