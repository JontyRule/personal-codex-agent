"""
Build embeddings locally and save them for deployment.
Run this script locally before deploying to avoid model download issues.
"""
import os
import sys
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.loader import load_markdown_files
from rag.splitter import split_markdown

def build_embeddings_locally():
    """Build embeddings locally and save for deployment"""
    print(" Building embeddings locally...")
    
    # Load documents
    doc_tuples = load_markdown_files()
    docs = [{"path": filepath, "content": content} for filepath, content in doc_tuples]
    print(f" Loaded {len(docs)} documents")
    
    if not docs:
        raise ValueError("No documents found in data/ directory")
    
    # Chunk all documents
    all_chunks = []
    for doc in docs:
        chunks = split_markdown(doc["content"], doc["path"])
        all_chunks.extend(chunks)
    
    print(f" Created {len(all_chunks)} chunks")
    
    # Load model locally
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f" Loaded model: all-MiniLM-L6-v2")
    
    # Create embeddings
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype('float32')
    
    print(f" Created embeddings shape: {embeddings.shape}")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Ensure cache directory exists
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save index and metadata
    index_path = os.path.join(cache_dir, "index.faiss")
    meta_path = os.path.join(cache_dir, "meta.json")
    embeddings_path = os.path.join(cache_dir, "embeddings.npy")
    
    # Save everything
    faiss.write_index(index, index_path)
    np.save(embeddings_path, embeddings)
    
    metadata = {
        "chunks": all_chunks,
        "count": len(all_chunks),
        "dimension": dimension,
        "model_name": "all-MiniLM-L6-v2",
        "embedding_dim": embeddings.shape[1]
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f" Index saved to {index_path}")
    print(f" Embeddings saved to {embeddings_path}")
    print(f" Metadata saved to {meta_path}")
    print(f" Ready for deployment! {len(all_chunks)} chunks embedded.")
    
    return metadata

if __name__ == "__main__":
    try:
        meta = build_embeddings_locally()
        print(f"\n Deployment ready! Index contains {meta['count']} chunks.")
        print("Now you can deploy without needing sentence-transformers on the server.")
    except Exception as e:
        print(f" Failed: {e}")
        sys.exit(1)
