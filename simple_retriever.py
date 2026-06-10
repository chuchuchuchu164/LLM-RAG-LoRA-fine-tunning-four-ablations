"""
Lightweight in-process retriever — a drop-in replacement for the ChromaDB
collection used in this project.

Why this exists: chromadb 1.5.x ships a Rust backend that triggers a Windows
access-violation crash (segfault) on this machine, and its default onnxruntime
embedder is also broken here. With only ~50 paper chunks we don't need a vector
database at all. This module embeds the SAME chunks with the SAME model
(all-MiniLM-L6-v2) and does exact cosine-similarity search in numpy, exposing
the small slice of the chromadb Collection API that the project uses
(.count() and .query(query_texts=..., n_results=...)).

Retrieval is therefore numerically equivalent to the previous ChromaDB setup
(which also used cosine distance over the same embeddings), with zero native
crash risk.
"""

import numpy as np

from paper_content import get_all_chunks, get_full_text

EMBED_MODEL = "all-MiniLM-L6-v2"


def sliding_window_chunks(text: str, chunk_size: int = 600, overlap: int = 120) -> list[dict]:
    """Identical chunking to create_rag.py so the corpus matches the old DB."""
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space > chunk_size // 2:
                chunk = chunk[:last_space]
                end = start + last_space
        chunks.append({
            "id": f"auto_{chunk_id:04d}",
            "section": "auto",
            "content": chunk.strip(),
        })
        chunk_id += 1
        start = end - overlap
    return chunks


def build_corpus() -> list[dict]:
    """curated section chunks + fine-grained sliding-window chunks."""
    curated = get_all_chunks()
    auto = sliding_window_chunks(get_full_text(), chunk_size=600, overlap=120)
    return curated + auto


class SimpleRetriever:
    """Minimal stand-in for a chromadb Collection (cosine similarity, numpy)."""

    def __init__(self, embed_model: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(embed_model)
        self.chunks = build_corpus()
        self.documents = [c["content"] for c in self.chunks]
        self.sections = [c.get("section", "unknown") for c in self.chunks]
        # Unit-normalized embeddings → dot product equals cosine similarity.
        self.embeddings = self.model.encode(
            self.documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def count(self) -> int:
        return len(self.chunks)

    def query(self, query_texts: list[str], n_results: int = 5) -> dict:
        """Return the same dict shape chromadb's Collection.query produces."""
        q = self.model.encode(
            query_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        documents, metadatas, distances = [], [], []
        for qi in q:
            sims = self.embeddings @ qi                  # cosine similarity
            top = np.argsort(-sims)[:n_results]
            documents.append([self.documents[i] for i in top])
            metadatas.append([{"section": self.sections[i]} for i in top])
            distances.append([float(1.0 - sims[i]) for i in top])  # cosine distance
        return {"documents": documents, "metadatas": metadatas, "distances": distances}


if __name__ == "__main__":
    r = SimpleRetriever()
    print(f"Corpus: {r.count()} chunks")
    res = r.query(["What convergence rate does MR.AsyncFL achieve?"], n_results=3)
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"\n[{meta['section']}] distance={dist:.4f}")
        print(doc[:200])
