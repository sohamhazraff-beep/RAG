"""Vector Store Module.

Implements a fast, zero-dependency, in-memory Vector Store powered by NumPy.
Performs exact cosine similarity search with score thresholding and metadata tracking.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from src.text_splitter import TextChunk


class VectorStore:
    """NumPy-backed vector store for semantic similarity search."""

    def __init__(self):
        self.chunks: list[TextChunk] = []
        self.embeddings: np.ndarray | None = None  # shape: (N, D)

    def is_empty(self) -> bool:
        return self.embeddings is None or len(self.chunks) == 0

    def add_chunks(self, chunks: Sequence[TextChunk], embeddings: np.ndarray) -> None:
        """Add text chunks and corresponding embedding vectors to the index.

        Args:
            chunks: Sequence of TextChunk objects.
            embeddings: 2D numpy array of shape (len(chunks), D).
        """
        if len(chunks) == 0:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        normalized_embs = (embeddings / norms).astype(np.float32)

        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = normalized_embs
            self.chunks = list(chunks)
        else:
            self.embeddings = np.vstack([self.embeddings, normalized_embs])
            self.chunks.extend(chunks)

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        score_threshold: float = 0.0,
    ) -> list[tuple[TextChunk, float]]:
        """Search for top_k most similar chunks to query_embedding.

        Returns list of (TextChunk, similarity_score) tuples sorted by score descending.
        """
        if self.is_empty():
            return []

        # Ensure query is 1D normalized float32
        q_vec = query_embedding.reshape(-1)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_norm_vec = (q_vec / q_norm).astype(np.float32)

        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self.embeddings, q_norm_vec)

        # Get top-k indices
        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]

        results: list[tuple[TextChunk, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= score_threshold:
                results.append((self.chunks[idx], score))

        return results

    def clear(self) -> None:
        """Reset the vector store."""
        self.chunks.clear()
        self.embeddings = None

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about indexed documents and vectors."""
        if self.is_empty():
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "unique_sources": [],
                "embedding_dimension": 0,
            }

        sources = sorted(list({c.source for c in self.chunks}))
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(sources),
            "unique_sources": sources,
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
        }

    def save(self, filepath: str | Path) -> None:
        """Persist vector index and chunks to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.is_empty():
            raise ValueError("Cannot save an empty vector store.")

        chunk_data = [
            {
                "text": c.text,
                "metadata": c.metadata,
                "chunk_index": c.chunk_index,
            }
            for c in self.chunks
        ]

        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            chunks_json=json.dumps(chunk_data),
        )

    def load(self, filepath: str | Path) -> None:
        """Load vector index and chunks from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        data = np.load(path, allow_pickle=False)
        self.embeddings = data["embeddings"]
        chunk_data = json.loads(str(data["chunks_json"]))
        self.chunks = [
            TextChunk(
                text=item["text"],
                metadata=item["metadata"],
                chunk_index=item["chunk_index"],
            )
            for item in chunk_data
        ]
