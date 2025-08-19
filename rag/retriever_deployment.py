"""
Deployment-ready retriever that uses pre-built embeddings
Uses proper embedding model for queries to match index quality
"""
import os
import json
import sys
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.logging_utils import get_logger

logger = get_logger("codex.retriever")

# Global model instance
_model = None

def get_embedding_model():
    """Get or load embedding model singleton"""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            logger.info("Loaded embedding model: all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Keep the simple fallback as backup
            _model = None
    return _model

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
    Fallback simple query embedding (lower quality)
    """
    words = query.lower().split()
    embedding = np.zeros(dimension, dtype='float32')
    
    for i, word in enumerate(words):
        if len(word) > 2:
            hash1 = hash(word) % dimension
            hash2 = hash(word[::-1]) % dimension
            weight = 1.0 / (i + 1)
            embedding[hash1] += weight
            embedding[hash2] += weight * 0.7
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding.reshape(1, -1)

def embed_query_proper(query: str) -> np.ndarray:
    """
    Proper query embedding using same model as index
    """
    model = get_embedding_model()
    if model is None:
        logger.warning("Using fallback embedding - quality will be reduced")
        return embed_query_simple(query)
    
    try:
        embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        return embedding.astype('float32')
    except Exception as e:
        logger.error(f"Failed to encode query: {e}")
        return embed_query_simple(query)

def retrieve_with_prebuilt(query: str, top_k: int = 4, prioritize_reflection: bool = False) -> Dict:
    """Retrieve relevant chunks using pre-built index"""
    try:
        index, metadata = load_prebuilt_index()
        
        # Use proper embedding model for query (same as used for index)
        query_embedding = embed_query_proper(query)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search with more candidates
        search_k = min(top_k * 3, len(metadata["chunks"]))
        scores, indices = index.search(query_embedding, search_k)
        
        results = []
        seen_sources = set()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
                
            chunk = metadata["chunks"][idx]
            source_path = chunk["source"]
            
            # Boost reflection documents if requested
            if prioritize_reflection and "self_reflection" in source_path.lower():
                score *= 1.3
            
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