from __future__ import annotations

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

def embed_query_simple(query: str) -> np.ndarray:
    """
    Simple query embedding without sentence-transformers for deployment
    Uses basic TF-IDF-like approach
    """
    try:
        words = query.lower().split()
        
        # Create a simple bag-of-words embedding
        embedding_dim = 384  # Match all-MiniLM-L6-v2 dimension
        embedding = np.zeros(embedding_dim, dtype='float32')
        
        for i, word in enumerate(words[:20]):  # Limit to first 20 words
            # Simple hash-based approach with position weighting
            hash_val = abs(hash(word)) % embedding_dim
            embedding[hash_val] += 1.0 / (i + 1)  # Give more weight to earlier words
        
        # Add some word-pair features for better matching
        for i in range(len(words) - 1):
            if i < 10:  # Limit bigrams
                bigram = words[i] + "_" + words[i + 1]
                hash_val = abs(hash(bigram)) % embedding_dim
                embedding[hash_val] += 0.5 / (i + 1)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding.reshape(1, -1)
        
    except Exception as e:
        logger.error(f"Failed to create query embedding: {e}")
        # Ultra-simple fallback
        embedding = np.random.random(384).astype('float32')
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.reshape(1, -1)


def load_index():
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


def retrieve(query: str, top_k: int = 4, prioritize_reflection: bool = False) -> Dict:
    """Retrieve relevant chunks using pre-built index"""
    try:
        index, metadata = load_index()

        # Create simple query embedding (no sentence-transformers needed)
        query_embedding = embed_query_simple(query)
        query_embedding = query_embedding.astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search
        search_k = min(top_k * 2, len(metadata["chunks"]))  # Get more candidates
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
                score *= 1.2

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
