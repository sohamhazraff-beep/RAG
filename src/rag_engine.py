"""RAG Engine Module.

Orchestrates the classic RAG pipeline:
1. Document ingestion and chunking
2. Embedding generation and vector indexing
3. Semantic retrieval (Cosine similarity)
4. Context-augmented prompt synthesis
5. Grounded LLM generation with streaming and source attribution
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Sequence
from google import genai
from google.genai import types

from src.document_loader import Document
from src.text_splitter import RecursiveCharacterTextSplitter, TextChunk
from src.embeddings import GeminiEmbeddingModel
from src.vector_store import VectorStore


@dataclass
class RetrievedSource:
    """Represents a retrieved source chunk with similarity score."""
    source: str
    chunk_index: int
    page: int | None
    similarity_score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Complete response from the RAG engine."""
    answer: str
    sources: list[RetrievedSource]
    query: str


DEFAULT_SYSTEM_PROMPT = """You are an accurate, honest, and helpful AI assistant specialized in Retrieval-Augmented Generation.
Your task is to answer the user's question using ONLY the provided retrieved context chunks.

Guidelines:
1. Rely strictly on facts directly mentioned in the retrieved context. Do NOT make up information or hallucinate facts not present in the context.
2. If the context does not contain enough information to answer the question, explicitly state: "Based on the provided documents, I could not find enough information to answer this question."
3. Cite the relevant source document name and page/chunk when presenting facts.
4. Keep answers concise, clear, and well-structured with bullet points or sections where appropriate.
"""


class RAGEngine:
    """Classic RAG pipeline coordinating Document Ingestion, Vector Search, and Gemini LLM."""

    def __init__(
        self,
        api_key: str,
        llm_model: str = "gemini-2.5-flash",
        embedding_model: str = "text-embedding-004",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("A valid Gemini API key is required to initialize RAGEngine.")

        self.api_key = api_key.strip()
        self.llm_model = llm_model
        self.system_prompt = system_prompt
        self.client = genai.Client(api_key=self.api_key)
        self.embedder = GeminiEmbeddingModel(api_key=self.api_key, model=embedding_model)
        self.vector_store = VectorStore()

    def set_models(self, llm_model: str | None = None, system_prompt: str | None = None) -> None:
        """Update active LLM model or system instructions."""
        if llm_model:
            self.llm_model = llm_model
        if system_prompt:
            self.system_prompt = system_prompt

    def ingest_documents(
        self,
        documents: Sequence[Document],
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        """Process, chunk, embed, and index a collection of documents."""
        if not documents:
            return {"status": "empty", "chunks": 0}

        if progress_callback:
            progress_callback(0.1, f"Splitting {len(documents)} documents into chunks...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            return {"status": "no_text_extracted", "chunks": 0}

        if progress_callback:
            progress_callback(0.3, f"Generating Gemini embeddings for {len(chunks)} chunks...")

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_documents(texts)

        if progress_callback:
            progress_callback(0.8, "Indexing vectors in in-memory vector store...")

        self.vector_store.add_chunks(chunks, embeddings)

        if progress_callback:
            progress_callback(1.0, f"Successfully indexed {len(chunks)} chunks!")

        return {
            "status": "success",
            "chunks": len(chunks),
            "documents": len(documents),
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: float = 0.0,
    ) -> list[RetrievedSource]:
        """Retrieve most relevant chunks for a user query."""
        if self.vector_store.is_empty():
            return []

        query_vector = self.embedder.embed_query(query)
        search_results = self.vector_store.similarity_search(
            query_embedding=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        sources: list[RetrievedSource] = []
        for chunk, score in search_results:
            sources.append(
                RetrievedSource(
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    page=chunk.page,
                    similarity_score=score,
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )
        return sources

    def _build_context_prompt(
        self,
        query: str,
        sources: list[RetrievedSource],
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Construct the prompt incorporating retrieved context and chat history."""
        context_blocks = []
        for i, s in enumerate(sources, start=1):
            page_info = f" | Page: {s.page}" if s.page else ""
            header = f"[Source #{i}: {s.source}{page_info} | Relevance Score: {s.similarity_score:.3f}]"
            context_blocks.append(f"{header}\n{s.text}")

        context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant context found in documents."

        # Include prior conversation turns if any
        history_str = ""
        if chat_history:
            recent_turns = chat_history[-6:]  # Keep last 3 exchanges
            formatted_turns = []
            for msg in recent_turns:
                role = "User" if msg.get("role") == "user" else "Assistant"
                formatted_turns.append(f"{role}: {msg.get('content', '')}")
            if formatted_turns:
                history_str = "\n\n--- RECENT CONVERSATION HISTORY ---\n" + "\n".join(formatted_turns)

        prompt = f"""--- RETRIEVED CONTEXT ---
{context_str}
{history_str}

--- USER QUESTION ---
{query}

Answer the user question based strictly on the retrieved context above. If citing sources, reference the source document names."""
        return prompt

    def generate_answer_stream(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: float = 0.0,
        temperature: float = 0.3,
        chat_history: list[dict[str, str]] | None = None,
    ) -> tuple[Generator[str, None, None], list[RetrievedSource]]:
        """Retrieve relevant context and stream the generated answer tokens."""
        sources = self.retrieve(query, top_k=top_k, score_threshold=score_threshold)
        prompt = self._build_context_prompt(query, sources, chat_history=chat_history)

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=self.system_prompt,
        )

        response_stream = self.client.models.generate_content_stream(
            model=self.llm_model,
            contents=prompt,
            config=config,
        )

        def text_generator() -> Generator[str, None, None]:
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        return text_generator(), sources

    def generate_answer(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: float = 0.0,
        temperature: float = 0.3,
        chat_history: list[dict[str, str]] | None = None,
    ) -> RAGResponse:
        """Synchronous non-streaming answer generation."""
        sources = self.retrieve(query, top_k=top_k, score_threshold=score_threshold)
        prompt = self._build_context_prompt(query, sources, chat_history=chat_history)

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=self.system_prompt,
        )

        response = self.client.models.generate_content(
            model=self.llm_model,
            contents=prompt,
            config=config,
        )

        answer_text = response.text if response.text else "No response generated."
        return RAGResponse(answer=answer_text, sources=sources, query=query)
