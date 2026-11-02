"""Document Loader Module.

Handles loading and parsing text from various document formats:
- PDF (.pdf)
- Plain Text (.txt)
- Markdown (.md)
- Microsoft Word (.docx)
- CSV (.csv)
"""

from __future__ import annotations
import io
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Sequence
import pypdf
import docx2txt


@dataclass
class Document:
    """Represents an ingested document or page."""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Clean null characters and strip trailing spaces
        self.content = self.content.replace("\x00", "").strip()


class DocumentLoader:
    """Multi-format document loader supporting files on disk or in-memory streams."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv"}

    @classmethod
    def load_from_stream(cls, stream: BinaryIO, filename: str) -> list[Document]:
        """Load documents from an in-memory binary stream (e.g. Streamlit UploadedFile)."""
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            return cls._load_pdf(stream, filename)
        elif suffix in {".txt", ".md"}:
            return cls._load_text(stream, filename)
        elif suffix == ".docx":
            return cls._load_docx(stream, filename)
        elif suffix == ".csv":
            return cls._load_csv(stream, filename)
        else:
            raise ValueError(f"Unsupported file format '{suffix}'. Supported: {cls.SUPPORTED_EXTENSIONS}")

    @classmethod
    def load_from_path(cls, file_path: str | Path) -> list[Document]:
        """Load documents from a local filesystem path."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        with open(path, "rb") as f:
            return cls.load_from_stream(f, path.name)

    @classmethod
    def _load_pdf(cls, stream: BinaryIO, filename: str) -> list[Document]:
        """Extract text from PDF page by page."""
        documents: list[Document] = []
        try:
            reader = pypdf.PdfReader(stream)
            total_pages = len(reader.pages)

            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            content=text,
                            metadata={
                                "source": filename,
                                "page": page_idx + 1,
                                "total_pages": total_pages,
                                "file_type": "pdf",
                                "char_count": len(text),
                            },
                        )
                    )
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF '{filename}': {e}") from e

        return documents

    @classmethod
    def _load_text(cls, stream: BinaryIO, filename: str) -> list[Document]:
        """Load plain text or markdown file."""
        raw_bytes = stream.read()
        for encoding in ("utf-8", "latin-1", "utf-16"):
            try:
                text = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw_bytes.decode("utf-8", errors="replace")

        return [
            Document(
                content=text,
                metadata={
                    "source": filename,
                    "page": 1,
                    "total_pages": 1,
                    "file_type": Path(filename).suffix.lstrip("."),
                    "char_count": len(text),
                },
            )
        ]

    @classmethod
    def _load_docx(cls, stream: BinaryIO, filename: str) -> list[Document]:
        """Load text from DOCX file."""
        try:
            text = docx2txt.process(stream) or ""
            return [
                Document(
                    content=text,
                    metadata={
                        "source": filename,
                        "page": 1,
                        "total_pages": 1,
                        "file_type": "docx",
                        "char_count": len(text),
                    },
                )
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to parse DOCX '{filename}': {e}") from e

    @classmethod
    def _load_csv(cls, stream: BinaryIO, filename: str) -> list[Document]:
        """Load CSV content as tabular text records."""
        try:
            raw_bytes = stream.read()
            text_data = raw_bytes.decode("utf-8", errors="replace")
            reader = csv.reader(io.StringIO(text_data))
            rows = list(reader)

            if not rows:
                return []

            header = rows[0]
            formatted_chunks = []
            for row_idx, row in enumerate(rows[1:], start=1):
                row_str = " | ".join(f"{h}: {val}" for h, val in zip(header, row))
                formatted_chunks.append(f"Row {row_idx}: {row_str}")

            full_text = "\n".join(formatted_chunks)
            return [
                Document(
                    content=full_text,
                    metadata={
                        "source": filename,
                        "page": 1,
                        "total_pages": 1,
                        "file_type": "csv",
                        "num_rows": len(rows) - 1,
                        "char_count": len(full_text),
                    },
                )
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to parse CSV '{filename}': {e}") from e
