from __future__ import annotations

import os
import sys
import json
from typing import List, Dict

# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.loader import load_markdown_files
from rag.splitter import split_markdown
from utils.logging_utils import get_logger


logger = get_logger("codex.index")

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(ROOT_DIR, "rag", "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.faiss")
META_PATH = os.path.join(CACHE_DIR, "meta.json")


def _get_embedding_model() -> SentenceTransformer:
    """Load lightweight sentence transformer model."""
    model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    logger.info(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def build_index() -> Dict:
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load docs
    md_files = load_markdown_files()
    if not md_files:
        raise RuntimeError("No markdown files found in data/. Add .md files and retry.")

    # Chunk
    all_chunks: List[Dict] = []
    for path, text in md_files:
        chunks = split_markdown(text, source_path=path)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No chunks generated from data/. Check your markdown content.")

    # Embed with lightweight model
    model = _get_embedding_model()
    texts = [c["text"] for c in all_chunks]
    logger.info(f"Embedding {len(texts)} chunks...")
    
    # Encode in batches to save memory
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        all_embeddings.append(embeddings)
    
    vecs = np.vstack(all_embeddings)
    
    # Build FAISS index
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for normalized embeddings
    index.add(vecs.astype('float32'))

    # Save index and metadata
    faiss.write_index(index, INDEX_PATH)

    meta = {
        "dimension": dim,
        "count": len(all_chunks),
        "model": os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "chunks": [
            {
                "text": c["text"],
                "source": os.path.relpath(c["source"], ROOT_DIR),
                "heading": c["heading"],
            }
            for c in all_chunks
        ],
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    logger.info(f"Index built: {INDEX_PATH} with {len(all_chunks)} vectors")
    return meta


def main() -> None:
    try:
        build_index()
    except Exception as e:
        logger.error(str(e))
        raise


if __name__ == "__main__":
    main()
