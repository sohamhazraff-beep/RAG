"""Text Splitter Module.

Implements recursive character text chunking with sliding window overlap
and metadata preservation for RAG pipelines.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence
from src.document_loader import Document


@dataclass
class TextChunk:
    """Represents an indexed chunk derived from a source document."""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def page(self) -> int | None:
        return self.metadata.get("page")


class RecursiveCharacterTextSplitter:
    """Recursively splits text using a hierarchical list of separators:
    paragraphs -> lines -> sentences -> words -> characters.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        min_chunk_size: int = 50,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.min_chunk_size = min_chunk_size

    def split_text(self, text: str) -> list[str]:
        """Split a string into chunks respecting size and overlap."""
        return self._split_text_recursive(text, self.separators)

    def _split_text_recursive(self, text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []

        # Find the best separator that appears in text
        separator = separators[-1]
        new_separators = []
        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)

        # Merge splits up to chunk_size with overlap
        current_chunk_parts: list[str] = []
        current_len = 0

        for part in splits:
            part_len = len(part) + (len(separator) if current_chunk_parts else 0)

            if current_len + part_len > self.chunk_size and current_chunk_parts:
                merged = separator.join(current_chunk_parts).strip()
                if len(merged) >= self.min_chunk_size:
                    final_chunks.append(merged)

                # Maintain overlap from current_chunk_parts
                while current_chunk_parts and current_len > self.chunk_overlap:
                    popped = current_chunk_parts.pop(0)
                    current_len -= len(popped) + (len(separator) if current_chunk_parts else 0)

            # If a single part exceeds chunk_size, split it with next separator level
            if len(part) > self.chunk_size:
                if new_separators:
                    sub_splits = self._split_text_recursive(part, new_separators)
                    final_chunks.extend(sub_splits)
                else:
                    final_chunks.append(part[: self.chunk_size])
            else:
                current_chunk_parts.append(part)
                current_len += len(part) + (len(separator) if len(current_chunk_parts) > 1 else 0)

        if current_chunk_parts:
            merged = separator.join(current_chunk_parts).strip()
            if len(merged) >= self.min_chunk_size:
                final_chunks.append(merged)

        return final_chunks

    def split_documents(self, documents: Sequence[Document]) -> list[TextChunk]:
        """Split a collection of Documents into TextChunks with metadata."""
        chunks: list[TextChunk] = []
        global_idx = 0

        for doc in documents:
            text_splits = self.split_text(doc.content)
            for local_idx, chunk_text in enumerate(text_splits):
                chunk_meta = dict(doc.metadata)
                chunk_meta["chunk_index"] = local_idx
                chunk_meta["global_index"] = global_idx
                chunk_meta["char_length"] = len(chunk_text)

                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        metadata=chunk_meta,
                        chunk_index=local_idx,
                    )
                )
                global_idx += 1

        return chunks
