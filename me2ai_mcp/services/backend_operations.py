"""
Backend operations for VectorStore service.

This module contains the implementations of the backend-specific operations
for each supported vector store type in the ME2AI MCP VectorStore service.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import os
import pickle
import asyncio
import uuid
import time

# Configure logging
logger = logging.getLogger("me2ai-mcp-vectorstore-backend")

# Default dimension for vector embeddings
DEFAULT_DIMENSION = 768
DEFAULT_DISTANCE = "cosine"

# Helper methods for matching
def match_metadata(metadata: Dict[str, Any], where: Dict[str, Any]) -> bool:
    """
    Check if metadata matches filter conditions.
    
    Args:
        metadata: Document metadata
        where: Metadata filter conditions
        
    Returns:
        bool: True if metadata matches conditions
    """
    for key, value in where.items():
        if key not in metadata or metadata[key] != value:
            return False
    return True

def match_document(text: str, where_document: Dict[str, Any]) -> bool:
    """
    Check if document text matches filter conditions.
    
    Args:
        text: Document text
        where_document: Document filter conditions
        
    Returns:
        bool: True if document matches conditions
    """
    # Currently only supports $contains operator
    if "$contains" in where_document:
        contains_value = where_document["$contains"]
        return contains_value in text
    return True

# ChromaDB operations
async def create_collection_chroma(client, name: str, embedding_function) -> Any:
    """
    Create a new ChromaDB collection.
    
    Args:
        client: ChromaDB client
        name: Collection name
        embedding_function: Function to use for embeddings
        
    Returns:
        Any: The created collection
    """
    collection = client.create_collection(
        name=name, 
        embedding_function=embedding_function
    )
    logger.info(f"Created ChromaDB collection: {name}")
    return collection

async def upsert_chroma(client, collections, collection_name: str, documents: List[str], 
                      metadatas: List[Dict[str, Any]], ids: List[str], embedding_function) -> None:
    """
    Add documents to ChromaDB collection.
    
    Args:
        client: ChromaDB client
        collections: Dict of available collections
        collection_name: Collection name
        documents: List of document texts
        metadatas: List of metadata dicts
        ids: List of document IDs
        embedding_function: Function to use for embeddings
    """
    # Get collection
    if collection_name not in collections:
        try:
            collection = client.get_collection(collection_name)
            logger.info(f"Using existing ChromaDB collection: {collection_name}")
        except Exception:
            collection = await create_collection_chroma(client, collection_name, embedding_function)
        collections[collection_name] = collection
    else:
        collection = collections[collection_name]
        
    # Add documents
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    logger.info(f"Upserted {len(documents)} documents to ChromaDB collection {collection_name}")

async def query_chroma(collections, collection_name: str, query: str, n_results: int,
                     where: Optional[Dict[str, Any]], where_document: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Query ChromaDB collection for similar documents.
    
    Args:
        collections: Dict of available collections
        collection_name: Collection name
        query: Query text
        n_results: Number of results to return
        where: Metadata filter
        where_document: Document content filter
        
    Returns:
        List[Dict[str, Any]]: Query results
    """
    # Get collection
    if collection_name not in collections:
        raise ValueError(f"Collection not found: {collection_name}")
    
    collection = collections[collection_name]
    
    # Query documents
    result = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        where_document=where_document
    )
    
    # Format results
    formatted_results = []
    for i in range(len(result["ids"][0])):
        formatted_results.append({
            "id": result["ids"][0][i],
            "text": result["documents"][0][i],
            "metadata": result["metadatas"][0][i],
            "distance": result["distances"][0][i] if "distances" in result else None
        })
        
    logger.info(f"Found {len(formatted_results)} results in ChromaDB collection {collection_name}")
    return formatted_results

async def delete_chroma(collections, collection_name: str, ids: Optional[List[str]],
                      where: Optional[Dict[str, Any]], where_document: Optional[Dict[str, Any]]) -> int:
    """
    Delete documents from ChromaDB collection.
    
    Args:
        collections: Dict of available collections
        collection_name: Collection name
        ids: List of document IDs to delete
        where: Metadata filter
        where_document: Document content filter
        
    Returns:
        int: Number of documents deleted
    """
    # Get collection
    if collection_name not in collections:
        raise ValueError(f"Collection not found: {collection_name}")
    
    collection = collections[collection_name]
    
    # Get count before deletion
    count_before = collection.count()
    
    # Delete documents
    if ids:
        collection.delete(ids=ids)
    elif where or where_document:
        collection.delete(where=where, where_document=where_document)
    else:
        # No filters specified, delete all
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
            
    # Get count after deletion
    count_after = collection.count()
    deleted_count = count_before - count_after
    
    logger.info(f"Deleted {deleted_count} documents from ChromaDB collection {collection_name}")
    return deleted_count

# FAISS operations
async def create_collection_faiss(faiss_dir: str, name: str) -> Dict[str, Any]:
    """
    Create a new FAISS collection.
    
    Args:
        faiss_dir: Directory for FAISS indices
        name: Collection name
        
    Returns:
        Dict[str, Any]: The created collection
    """
    import faiss
    
    # Create new index
    dimension = DEFAULT_DIMENSION
    index = faiss.IndexFlatL2(dimension)
    data = {
        "ids": [],
        "texts": [],
        "metadatas": []
    }
    
    # Save index and data
    index_path = os.path.join(faiss_dir, f"{name}.index")
    data_path = os.path.join(faiss_dir, f"{name}.pickle")
    
    faiss.write_index(index, index_path)
    with open(data_path, "wb") as f:
        pickle.dump(data, f)
    
    collection = {
        "index": index,
        "data": data
    }
    
    logger.info(f"Created FAISS collection: {name}")
    return collection

async def upsert_faiss(faiss_dir: str, collections: Dict[str, Any], collection_name: str, 
                     documents: List[str], metadatas: List[Dict[str, Any]], 
                     ids: List[str], embedding_function) -> None:
    """
    Add documents to FAISS collection.
    
    Args:
        faiss_dir: Directory for FAISS indices
        collections: Dict of available collections
        collection_name: Collection name
        documents: List of document texts
        metadatas: List of metadata dicts
        ids: List of document IDs
        embedding_function: Function to use for embeddings
    """
    import faiss
    import numpy as np
    
    # Get collection
    if collection_name not in collections:
        collections[collection_name] = await create_collection_faiss(faiss_dir, collection_name)
        
    collection = collections[collection_name]
    index = collection["index"]
    data = collection["data"]
    
    # Get embeddings
    embeddings = np.array(embedding_function(documents), dtype=np.float32)
    
    # Add to index
    index.add(embeddings)
    
    # Update data
    data["ids"].extend(ids)
    data["texts"].extend(documents)
    data["metadatas"].extend(metadatas)
    
    # Save index and data
    index_path = os.path.join(faiss_dir, f"{collection_name}.index")
    data_path = os.path.join(faiss_dir, f"{collection_name}.pickle")
    
    faiss.write_index(index, index_path)
    with open(data_path, "wb") as f:
        pickle.dump(data, f)
        
    logger.info(f"Upserted {len(documents)} documents to FAISS collection {collection_name}")

async def query_faiss(collections: Dict[str, Any], collection_name: str, query: str, 
                    n_results: int, where: Optional[Dict[str, Any]], 
                    where_document: Optional[Dict[str, Any]], embedding_function) -> List[Dict[str, Any]]:
    """
    Query FAISS collection for similar documents.
    
    Args:
        collections: Dict of available collections
        collection_name: Collection name
        query: Query text
        n_results: Number of results to return
        where: Metadata filter
        where_document: Document content filter
        embedding_function: Function to use for embeddings
        
    Returns:
        List[Dict[str, Any]]: Query results
    """
    import numpy as np
    
    # Get collection
    if collection_name not in collections:
        raise ValueError(f"Collection not found: {collection_name}")
    
    collection = collections[collection_name]
    index = collection["index"]
    data = collection["data"]
    
    # Get query embedding
    query_embedding = np.array([embedding_function(query)], dtype=np.float32)
    
    # Search index
    distances, indices = index.search(query_embedding, n_results)
    
    # Format results
    formatted_results = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0 and idx < len(data["ids"]):
            # Apply metadata filter
            if where and not match_metadata(data["metadatas"][idx], where):
                continue
                
            # Apply document filter
            if where_document and not match_document(data["texts"][idx], where_document):
                continue
                
            formatted_results.append({
                "id": data["ids"][idx],
                "text": data["texts"][idx],
                "metadata": data["metadatas"][idx],
                "distance": float(distances[0][i])
            })
            
    logger.info(f"Found {len(formatted_results)} results in FAISS collection {collection_name}")
    return formatted_results

async def delete_faiss(faiss_dir: str, collections: Dict[str, Any], collection_name: str,
                     ids: Optional[List[str]], where: Optional[Dict[str, Any]], 
                     where_document: Optional[Dict[str, Any]], embedding_function) -> int:
    """
    Delete documents from FAISS collection.
    
    Args:
        faiss_dir: Directory for FAISS indices
        collections: Dict of available collections
        collection_name: Collection name
        ids: List of document IDs to delete
        where: Metadata filter
        where_document: Document content filter
        embedding_function: Function to use for embeddings
        
    Returns:
        int: Number of documents deleted
    """
    import faiss
    import numpy as np
    
    # Get collection
    if collection_name not in collections:
        raise ValueError(f"Collection not found: {collection_name}")
    
    collection = collections[collection_name]
    data = collection["data"]
    
    # Get count before deletion
    count_before = len(data["ids"])
    
    # Build new data without deleted documents
    new_ids = []
    new_texts = []
    new_metadatas = []
    
    for i, doc_id in enumerate(data["ids"]):
        keep = True
        
        # Check if ID is in deletion list
        if ids and doc_id in ids:
            keep = False
            
        # Check if metadata matches deletion filter
        if keep and where and match_metadata(data["metadatas"][i], where):
            keep = False
            
        # Check if document content matches deletion filter
        if keep and where_document and match_document(data["texts"][i], where_document):
            keep = False
            
        if keep:
            new_ids.append(doc_id)
            new_texts.append(data["texts"][i])
            new_metadatas.append(data["metadatas"][i])
    
    # If nothing changed, return 0
    if len(new_ids) == count_before:
        return 0
        
    # Recreate index with remaining documents
    dimension = DEFAULT_DIMENSION
    new_index = faiss.IndexFlatL2(dimension)
    
    if new_ids:  # Only add vectors if we have documents left
        # Get embeddings for remaining documents
        embeddings = np.array(embedding_function(new_texts), dtype=np.float32)
        new_index.add(embeddings)
        
    # Update collection
    collection["index"] = new_index
    collection["data"] = {
        "ids": new_ids,
        "texts": new_texts,
        "metadatas": new_metadatas
    }
    
    # Save updated index and data
    index_path = os.path.join(faiss_dir, f"{collection_name}.index")
    data_path = os.path.join(faiss_dir, f"{collection_name}.pickle")
    
    faiss.write_index(new_index, index_path)
    with open(data_path, "wb") as f:
        pickle.dump(collection["data"], f)
        
    deleted_count = count_before - len(new_ids)
    logger.info(f"Deleted {deleted_count} documents from FAISS collection {collection_name}")
    return deleted_count

# Qdrant operations
async def create_collection_qdrant(client, name: str) -> None:
    """
    Create a new Qdrant collection.
    
    Args:
        client: Qdrant client
        name: Collection name
    """
    from qdrant_client.http import models
    
    # Create collection
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=DEFAULT_DIMENSION,
            distance=DEFAULT_DISTANCE
        )
    )
    
    logger.info(f"Created Qdrant collection: {name}")

async def upsert_qdrant(client, collection_name: str, documents: List[str], 
                      metadatas: List[Dict[str, Any]], ids: List[str], embedding_function) -> None:
    """
    Add documents to Qdrant collection.
    
    Args:
        client: Qdrant client
        collection_name: Collection name
        documents: List of document texts
        metadatas: List of metadata dicts
        ids: List of document IDs
        embedding_function: Function to use for embeddings
    """
    from qdrant_client.http import models
    
    # Get embeddings
    embeddings = embedding_function(documents)
    
    # Prepare points
    points = []
    for i, doc_id in enumerate(ids):
        # Add text to metadata
        metadata = dict(metadatas[i])
        metadata["text"] = documents[i]
        
        points.append(models.PointStruct(
            id=doc_id,
            vector=embeddings[i],
            payload=metadata
        ))
        
    # Upsert points
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    logger.info(f"Upserted {len(documents)} documents to Qdrant collection {collection_name}")

async def query_qdrant(client, collection_name: str, query: str, n_results: int,
                    where: Optional[Dict[str, Any]], where_document: Optional[Dict[str, Any]], 
                    embedding_function) -> List[Dict[str, Any]]:
    """
    Query Qdrant collection for similar documents.
    
    Args:
        client: Qdrant client
        collection_name: Collection name
        query: Query text
        n_results: Number of results to return
        where: Metadata filter
        where_document: Document content filter
        embedding_function: Function to use for embeddings
        
    Returns:
        List[Dict[str, Any]]: Query results
    """
    from qdrant_client.http import models
    
    # Get query embedding
    query_embedding = embedding_function(query)
    
    # Prepare filter
    filter_obj = None
    if where:
        conditions = []
        for key, value in where.items():
            conditions.append(models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value)
            ))
            
        if conditions:
            filter_obj = models.Filter(must=conditions)
            
    # Search
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=n_results,
        filter=filter_obj
    )
    
    # Format results
    formatted_results = []
    for result in results:
        # Extract text from payload
        metadata = result.payload if result.payload else {}
        text = metadata.pop("text", "") if metadata else ""
        
        # Apply document filter if needed
        if where_document and not match_document(text, where_document):
            continue
            
        formatted_results.append({
            "id": result.id,
            "text": text,
            "metadata": metadata,
            "distance": result.score
        })
        
    logger.info(f"Found {len(formatted_results)} results in Qdrant collection {collection_name}")
    return formatted_results

async def delete_qdrant(client, collection_name: str, ids: Optional[List[str]],
                     where: Optional[Dict[str, Any]], where_document: Optional[Dict[str, Any]]) -> int:
    """
    Delete documents from Qdrant collection.
    
    Args:
        client: Qdrant client
        collection_name: Collection name
        ids: List of document IDs to delete
        where: Metadata filter
        where_document: Document content filter
        
    Returns:
        int: Number of documents deleted
    """
    from qdrant_client.http import models
    
    # Get count before deletion
    count_before = client.count(collection_name=collection_name).count
    
    # Delete by IDs
    if ids:
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=ids)
        )
        
    # Delete by filter
    elif where:
        conditions = []
        for key, value in where.items():
            conditions.append(models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value)
            ))
            
        if conditions:
            client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=conditions)
                )
            )
            
    # Handle document content filter - need to do a search first then delete IDs
    elif where_document:
        # This is inefficient but necessary for document content filtering
        # Get all documents
        results = client.scroll(
            collection_name=collection_name,
            limit=1000,  # Use a reasonable batch size
            with_payload=True
        )[0]
        
        # Filter document IDs that match the content filter
        ids_to_delete = []
        for point in results:
            if "text" in point.payload and match_document(point.payload["text"], where_document):
                ids_to_delete.append(point.id)
                
        # Delete filtered IDs
        if ids_to_delete:
            client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=ids_to_delete)
            )
    
    # Get count after deletion
    count_after = client.count(collection_name=collection_name).count
    deleted_count = count_before - count_after
    
    logger.info(f"Deleted {deleted_count} documents from Qdrant collection {collection_name}")
    return deleted_count

# Pinecone operations
async def upsert_pinecone(index, collection_name: str, documents: List[str], 
                       metadatas: List[Dict[str, Any]], ids: List[str], embedding_function) -> None:
    """
    Add documents to Pinecone index.
    
    Args:
        index: Pinecone index
        collection_name: Collection name (namespace in Pinecone)
        documents: List of document texts
        metadatas: List of metadata dicts
        ids: List of document IDs
        embedding_function: Function to use for embeddings
    """
    # Get embeddings
    embeddings = embedding_function(documents)
    
    # Prepare records
    records = []
    for i, doc_id in enumerate(ids):
        # Add text to metadata
        metadata = dict(metadatas[i])
        metadata["text"] = documents[i]
        
        records.append({
            "id": doc_id,
            "values": embeddings[i],
            "metadata": metadata
        })
        
    # Upsert in batches (Pinecone has limits)
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        index.upsert(vectors=batch, namespace=collection_name)
        
    logger.info(f"Upserted {len(documents)} documents to Pinecone namespace {collection_name}")

async def query_pinecone(index, collection_name: str, query: str, n_results: int,
                     where: Optional[Dict[str, Any]], where_document: Optional[Dict[str, Any]], 
                     embedding_function) -> List[Dict[str, Any]]:
    """
    Query Pinecone index for similar documents.
    
    Args:
        index: Pinecone index
        collection_name: Collection name (namespace in Pinecone)
        query: Query text
        n_results: Number of results to return
        where: Metadata filter
        where_document: Document content filter
        embedding_function: Function to use for embeddings
        
    Returns:
        List[Dict[str, Any]]: Query results
    """
    # Get query embedding
    query_embedding = embedding_function(query)
    
    # Prepare filter
    filter_dict = None
    if where:
        filter_dict = where
        
    # Search
    results = index.query(
        vector=query_embedding,
        namespace=collection_name,
        top_k=n_results,
        include_metadata=True,
        filter=filter_dict
    )
    
    # Format results
    formatted_results = []
    for match in results["matches"]:
        # Extract text from metadata
        metadata = match["metadata"] if "metadata" in match else {}
        text = metadata.pop("text", "") if metadata else ""
        
        # Apply document filter if needed
        if where_document and not match_document(text, where_document):
            continue
            
        formatted_results.append({
            "id": match["id"],
            "text": text,
            "metadata": metadata,
            "distance": match["score"]
        })
        
    logger.info(f"Found {len(formatted_results)} results in Pinecone namespace {collection_name}")
    return formatted_results

async def delete_pinecone(index, collection_name: str, ids: Optional[List[str]],
                       where: Optional[Dict[str, Any]], where_document: Optional[Dict[str, Any]]) -> int:
    """
    Delete documents from Pinecone index.
    
    Args:
        index: Pinecone index
        collection_name: Collection name (namespace in Pinecone)
        ids: List of document IDs to delete
        where: Metadata filter
        where_document: Document content filter
        
    Returns:
        int: Number of documents deleted
    """
    # Get count before deletion
    stats = index.describe_index_stats()
    count_before = 0
    if "namespaces" in stats and collection_name in stats["namespaces"]:
        count_before = stats["namespaces"][collection_name]["vector_count"]
        
    # Delete by IDs
    if ids:
        index.delete(ids=ids, namespace=collection_name)
        
    # Delete by filter
    elif where:
        # Fetch matching IDs first (Pinecone doesn't support delete by filter)
        # This is inefficient for large collections
        query_response = index.query(
            vector=[0.0] * DEFAULT_DIMENSION,  # Dummy vector
            namespace=collection_name,
            top_k=10000,  # Adjust as needed
            include_metadata=False,
            filter=where
        )
        
        matched_ids = [match["id"] for match in query_response["matches"]]
        
        if matched_ids:
            index.delete(ids=matched_ids, namespace=collection_name)
            
    # Handle document content filter
    elif where_document:
        # This requires fetching all vectors, very inefficient
        # Query all vectors in namespace
        query_response = index.query(
            vector=[0.0] * DEFAULT_DIMENSION,  # Dummy vector
            namespace=collection_name,
            top_k=10000,  # Adjust as needed
            include_metadata=True
        )
        
        # Filter document IDs that match the content filter
        ids_to_delete = []
        for match in query_response["matches"]:
            metadata = match.get("metadata", {})
            text = metadata.get("text", "")
            if text and match_document(text, where_document):
                ids_to_delete.append(match["id"])
                
        # Delete filtered IDs
        if ids_to_delete:
            index.delete(ids=ids_to_delete, namespace=collection_name)
    
    # Get count after deletion
    stats = index.describe_index_stats()
    count_after = 0
    if "namespaces" in stats and collection_name in stats["namespaces"]:
        count_after = stats["namespaces"][collection_name]["vector_count"]
        
    deleted_count = count_before - count_after
    logger.info(f"Deleted {deleted_count} documents from Pinecone namespace {collection_name}")
    return deleted_count
