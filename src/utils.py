"""Utility functions for the Streamlit RAG application."""

from __future__ import annotations
from pathlib import Path
from typing import Sequence
from google import genai
from src.document_loader import Document, DocumentLoader


def validate_gemini_api_key(api_key: str) -> tuple[bool, str | None]:
    """Test whether a given Gemini API key is valid by querying the models list.

    Returns:
        (True, None) if key is valid.
        (False, error_message) if invalid.
    """
    if not api_key or not api_key.strip():
        return False, "API key cannot be empty."

    key = api_key.strip()
    try:
        client = genai.Client(api_key=key)
        # Attempt lightweight call to verify key
        for _ in client.models.list():
            break
        return True, None
    except Exception as e:
        error_msg = str(e)
        if "API key not valid" in error_msg or "INVALID_ARGUMENT" in error_msg:
            return False, "Invalid API key. Please check your Google AI Studio key."
        elif "PERMISSION_DENIED" in error_msg:
            return False, "Permission denied for this API key."
        elif "RESOURCE_EXHAUSTED" in error_msg:
            return False, "API quota exhausted. Please check your Gemini account limits."
        return False, f"Validation failed: {error_msg[:120]}"


def load_sample_documents(sample_dir: str | Path | None = None) -> list[Document]:
    """Load bundled sample documents from data/sample_documents."""
    if sample_dir is None:
        sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample_documents"
    else:
        sample_dir = Path(sample_dir)

    if not sample_dir.exists():
        return []

    docs: list[Document] = []
    for file_path in sorted(sample_dir.glob("*")):
        if file_path.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS:
            try:
                loaded = DocumentLoader.load_from_path(file_path)
                docs.extend(loaded)
            except Exception as e:
                print(f"Warning: Failed to load sample file {file_path.name}: {e}")
    return docs


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (approx. 4 characters per token)."""
    return max(1, len(text) // 4)


def format_file_size(num_bytes: int) -> str:
    """Format bytes into readable size strings."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"
