"""Embeddings Module.

Wraps Google Gemini embedding models (`text-embedding-004`) with batching,
type conversion, and error handling.
"""

from __future__ import annotations
import time
from typing import Sequence
import numpy as np
from google import genai
from google.genai.errors import APIError


class GeminiEmbeddingModel:
    """Handles text embedding using Google Gemini API."""

    DEFAULT_MODEL = "text-embedding-004"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key or not api_key.strip():
            raise ValueError("A valid Gemini API key is required.")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.client = genai.Client(api_key=self.api_key)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns 1D float32 numpy array."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        response = self.client.models.embed_content(
            model=self.model,
            contents=[query.strip()],
        )

        if not response.embeddings or not response.embeddings[0].values:
            raise RuntimeError("Gemini API returned an empty embedding for query.")

        vector = np.array(response.embeddings[0].values, dtype=np.float32)
        return vector

    def embed_documents(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed multiple text chunks in batches.

        Returns 2D float32 numpy array with shape (num_texts, embedding_dim).
        """
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        cleaned_texts = [t.strip() if t.strip() else " " for t in texts]
        all_embeddings: list[list[float]] = []

        total = len(cleaned_texts)
        for i in range(0, total, batch_size):
            batch = cleaned_texts[i : i + batch_size]
            retries = 3
            backoff = 1.0

            for attempt in range(retries):
                try:
                    response = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                    )
                    if response.embeddings:
                        for emb in response.embeddings:
                            all_embeddings.append(emb.values)
                    break
                except APIError as e:
                    if attempt < retries - 1 and ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                        time.sleep(backoff)
                        backoff *= 2
                    else:
                        raise RuntimeError(f"Gemini API embedding error: {e}") from e
                except Exception as e:
                    if attempt < retries - 1:
                        time.sleep(backoff)
                        backoff *= 2
                    else:
                        raise RuntimeError(f"Embedding failed: {e}") from e

        if len(all_embeddings) != total:
            raise RuntimeError(
                f"Embedding count mismatch: expected {total}, got {len(all_embeddings)}"
            )

        return np.array(all_embeddings, dtype=np.float32)
