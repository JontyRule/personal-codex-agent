"""
Deployment-ready retriever that uses pre-built embeddings
No sentence-transformers dependency required on server
"""
import os
import json
import sys
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np

from utils.logging_utils import get_logger

logger = get_logger("codex.retriever")

def load_prebuilt_index():
    """Load pre-built FAISS index and metadata"""
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    index_path = os.path.join(cache_dir, "index.faiss")
    meta_path = os.path.join(cache_dir, "meta.json")
    
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Pre-built index not found. Run 'python scripts/build_embeddings_local.py' locally first."
        )
    
    index = faiss.read_index(index_path)
    
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded pre-built index with {metadata['count']} chunks")
    return index, metadata

def embed_query_simple(query: str, dimension: int = 384) -> np.ndarray:
    """
    Simple query embedding without sentence-transformers
    This is a basic fallback approach for deployment
    """
    # Simple tokenization and basic embedding
    words = query.lower().split()
    
    # Create a simple TF-based embedding
    embedding = np.zeros(dimension, dtype='float32')
    
    # Basic word hashing approach
    for i, word in enumerate(words):
        if len(word) > 2:  # Skip very short words
            # Use multiple hash functions for better distribution
            hash1 = hash(word) % dimension
            hash2 = hash(word[::-1]) % dimension  # Reverse word hash
            hash3 = hash(word + str(len(word))) % dimension  # Length-based hash
            
            # Weight words by position (earlier words get higher weight)
            weight = 1.0 / (i + 1)
            
            embedding[hash1] += weight
            embedding[hash2] += weight * 0.7
            embedding[hash3] += weight * 0.5
    
    # Add query length factor
    length_factor = min(len(words) / 10.0, 1.0)
    embedding *= (0.5 + length_factor)
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding.reshape(1, -1)

def retrieve_with_prebuilt(query: str, top_k: int = 4, prioritize_reflection: bool = False) -> Dict:
    """Retrieve relevant chunks using pre-built index"""
    try:
        index, metadata = load_prebuilt_index()
        
        # Get query embedding (simplified version for deployment)
        dimension = metadata.get("dimension", 384)
        query_embedding = embed_query_simple(query, dimension)
        query_embedding = query_embedding.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        search_k = min(top_k * 3, len(metadata["chunks"]))  # Get more candidates for filtering
        scores, indices = index.search(query_embedding, search_k)
        
        results = []
        seen_sources = set()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for not found
                continue
                
            chunk = metadata["chunks"][idx]
            source_path = chunk["source"]
            
            # Boost reflection documents if requested
            if prioritize_reflection and "self_reflection" in source_path.lower():
                score *= 1.3
            
            # Simple keyword matching boost
            query_words = set(query.lower().split())
            chunk_words = set(chunk["text"].lower().split())
            common_words = query_words.intersection(chunk_words)
            if common_words:
                keyword_boost = min(len(common_words) * 0.1, 0.3)
                score *= (1.0 + keyword_boost)
            
            # Limit results per source for diversity
            source_count = sum(1 for r in results if r["source"] == source_path)
            if source_count >= 2:
                continue
                
            results.append({
                "text": chunk["text"],
                "source": source_path,
                "heading": chunk.get("heading", ""),
                "score": float(score)
            })
            
            seen_sources.add(source_path)
            
            if len(results) >= top_k:
                break
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "results": results,
            "count": len(results),
            "max_score": max([r["score"] for r in results]) if results else 0.0,
            "avg_score": sum([r["score"] for r in results]) / len(results) if results else 0.0,
        }
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {
            "results": [],
            "count": 0,
            "max_score": 0.0,
            "avg_score": 0.0,
            "error": str(e)
        }

# Backward compatibility alias
retrieve = retrieve_with_prebuilt