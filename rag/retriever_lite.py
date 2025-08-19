from __future__ import annotations

import os
import sys
import json
from typing import List, Dict, Any

# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss  # type: ignore
import numpy as np  # type: ignore
from sentence_transformers import SentenceTransformer

from utils.logging_utils import get_logger


logger = get_logger("codex.retriever")

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(ROOT_DIR, "rag", "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.faiss")
META_PATH = os.path.join(CACHE_DIR, "meta.json")

# Cache the model to avoid reloading
_embedding_model = None


def _get_embedding_model() -> SentenceTransformer:
    """Get cached embedding model."""
    global _embedding_model
    if _embedding_model is None:
        model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def _embed_query(query: str) -> np.ndarray:
    """Embed a single query efficiently."""
    model = _get_embedding_model()
    embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return embedding.astype('float32')


def _load_index_and_meta():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        raise FileNotFoundError(
            "Missing index. Build it via 'python rag/build_index_lite.py' or the UI button."
        )
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


def retrieve(query: str, top_k: int = 4, prioritize_reflection: bool = False) -> Dict[str, Any]:
    index, meta = _load_index_and_meta()
    q = _embed_query(query)
    scores, idxs = index.search(q, max(top_k * 3, top_k))  # search wider, then filter
    idxs = idxs[0]
    scores = scores[0]

    chunks = meta["chunks"]

    # Build candidates with score and apply optional reflection prioritization
    candidates = []
    for i, s in zip(idxs.tolist(), scores.tolist()):
        if i < 0 or i >= len(chunks):
            continue
        c = chunks[i]
        boost = 1.2 if (prioritize_reflection and c["source"].endswith("self_reflection.md")) else 1.0
        candidates.append({
            "text": c["text"],
            "source": c["source"],
            "heading": c["heading"],
            "score": float(s) * boost,
        })

    # Sort by boosted score and take top_k unique by source+heading to add diversity
    seen_keys = set()
    results = []
    for item in sorted(candidates, key=lambda x: x["score"], reverse=True):
        key = (item["source"], item["heading"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        results.append(item)
        if len(results) >= top_k:
            break

    max_score = max((r["score"] for r in results), default=0.0)
    avg_score = sum((r["score"] for r in results), 0.0) / max(len(results), 1)

    return {
        "results": results,
        "max_score": max_score,
        "avg_score": avg_score,
        "count": len(results),
    }
