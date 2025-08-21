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

from utils.loader import load_markdown_files
from utils.logging_utils import get_logger
from rag.splitter import chunk_markdown


logger = get_logger("codex.build_index")


def load_embedding_model():
    """Load embedding model with error handling for deployment"""
    model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    try:
        # Force CPU usage and avoid meta tensor issues
        model = SentenceTransformer(model_name, device='cpu')
        logger.info(f"Loaded embedding model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        # Fallback to a more basic model
        try:
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')
            logger.info("Loaded fallback model: paraphrase-MiniLM-L6-v2")
            return model
        except Exception as e2:
            logger.error(f"Failed to load fallback model: {e2}")
            raise RuntimeError(f"Could not load any embedding model. Original error: {e}")


def build_index() -> Dict:
    """Build FAISS index from markdown files in data/ directory"""
    logger.info("Building index...")

    # Ensure cache directory exists
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Load documents
    docs = load_markdown_files()
    logger.info(f"Loaded {len(docs)} documents")

    if not docs:
        raise ValueError("No documents found in data/ directory")

    # Chunk all documents
    all_chunks = []
    for doc in docs:
        chunks = chunk_markdown(doc["content"], doc["path"])
        all_chunks.extend(chunks)

    logger.info(f"Created {len(all_chunks)} chunks")

    if not all_chunks:
        raise ValueError("No chunks created from documents")

    # Load embedding model
    model = load_embedding_model()

    # Create embeddings in batches to avoid memory issues
    batch_size = 32
    embeddings = []

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        batch_texts = [chunk["text"] for chunk in batch]

        try:
            batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
        except Exception as e:
            logger.error(f"Failed to encode batch {i//batch_size + 1}: {e}")
            raise

    embeddings = np.array(embeddings).astype('float32')
    logger.info(f"Created embeddings shape: {embeddings.shape}")

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save index and metadata
    index_path = os.path.join(cache_dir, "index.faiss")
    meta_path = os.path.join(cache_dir, "meta.json")

    faiss.write_index(index, index_path)

    metadata = {
        "chunks": all_chunks,
        "count": len(all_chunks),
        "dimension": dimension,
        "model": model.get_sentence_embedding_dimension()
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Index saved to {index_path}")
    logger.info(f"Metadata saved to {meta_path}")

    return metadata


if __name__ == "__main__":
    try:
        meta = build_index()
        print(f" Index built successfully with {meta['count']} chunks")
    except Exception as e:
        print(f" Failed to build index: {e}")
        sys.exit(1)
