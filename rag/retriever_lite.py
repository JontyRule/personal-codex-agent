from __future__ import annotations

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
        model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        try:
            # Force CPU usage and avoid meta tensor issues
            _model = SentenceTransformer(model_name, device='cpu')
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            # Fallback to a more basic model
            try:
                _model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')
                logger.info("Loaded fallback model: paraphrase-MiniLM-L6-v2")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise RuntimeError(f"Could not load any embedding model. Original error: {e}")
    return _model


def load_index():
    """Load FAISS index and metadata"""
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    index_path = os.path.join(cache_dir, "index.faiss")
    meta_path = os.path.join(cache_dir, "meta.json")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Index not found. Build it via 'python rag/build_index_lite.py' or the UI button."
        )

    index = faiss.read_index(index_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def retrieve(query: str, top_k: int = 4, prioritize_reflection: bool = False) -> Dict:
    """Retrieve relevant chunks for a query"""
    try:
        index, metadata = load_index()
        model = get_embedding_model()

        # Encode query
        query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
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
