"""
Create a ChromaDB vector store from the MR.AsyncFL paper content.
Run this once to build the RAG database.

Usage:
    python create_rag.py
"""

import chromadb
from chromadb.utils import embedding_functions
from paper_content import get_all_chunks, get_full_text
import re


DB_PATH = "./chroma_db"
COLLECTION_NAME = "mrasyncfl_paper"

EMBED_MODEL = "all-MiniLM-L6-v2"  # fast, good quality for technical text


def sliding_window_chunks(text: str, chunk_size: int = 600, overlap: int = 100) -> list[dict]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Avoid cutting in the middle of a word
        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space > chunk_size // 2:
                chunk = chunk[:last_space]
                end = start + last_space
        chunks.append({
            "id": f"auto_{chunk_id:04d}",
            "section": "auto",
            "content": chunk.strip()
        })
        chunk_id += 1
        start = end - overlap
    return chunks


def build_database():
    """Embed paper chunks and store in ChromaDB."""
    print(f"Initializing ChromaDB at {DB_PATH} ...")
    client = chromadb.PersistentClient(path=DB_PATH)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # Drop and recreate collection for a clean build
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Use the curated section chunks from paper_content.py
    curated_chunks = get_all_chunks()

    # Also add fine-grained sliding-window chunks from the full text
    # so we don't miss any detail
    full_text = get_full_text()
    auto_chunks = sliding_window_chunks(full_text, chunk_size=600, overlap=120)

    all_chunks = curated_chunks + auto_chunks
    print(f"Total chunks to embed: {len(all_chunks)}")

    ids = [c["id"] for c in all_chunks]
    documents = [c["content"] for c in all_chunks]
    metadatas = [{"section": c["section"]} for c in all_chunks]

    # Upsert in batches of 50
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
        print(f"  Embedded {min(i+batch_size, len(all_chunks))}/{len(all_chunks)} chunks")

    print(f"\nRAG database built. Collection '{COLLECTION_NAME}' has {collection.count()} entries.")
    return collection


if __name__ == "__main__":
    build_database()
    print("Done. Run query_rag.py to test retrieval.")
