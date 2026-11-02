"""Classic RAG Application with Streamlit and Google Gemini.

Allows users to upload documents (PDF, TXT, MD, DOCX, CSV), input their Gemini API key,
index content into an in-memory vector store, and chat with grounded generation and source citations.
"""

from __future__ import annotations
import os
from typing import Any
import streamlit as st

from src.document_loader import Document, DocumentLoader
from src.rag_engine import RAGEngine, DEFAULT_SYSTEM_PROMPT
from src.utils import validate_gemini_api_key, load_sample_documents, format_file_size

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gemini Classic RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False

if "last_tested_key" not in st.session_state:
    st.session_state.last_tested_key = ""


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def get_active_engine(api_key: str, llm_model: str, system_prompt: str) -> RAGEngine | None:
    """Get or initialize RAGEngine instance."""
    if not api_key:
        return None

    if st.session_state.rag_engine is None:
        try:
            st.session_state.rag_engine = RAGEngine(
                api_key=api_key,
                llm_model=llm_model,
                system_prompt=system_prompt,
            )
        except Exception as e:
            st.sidebar.error(f"Failed to initialize RAG Engine: {e}")
            return None
    else:
        st.session_state.rag_engine.set_models(
            llm_model=llm_model,
            system_prompt=system_prompt,
        )
    return st.session_state.rag_engine


# ---------------------------------------------------------
# Sidebar: Configuration & Ingestion
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚡ Gemini Classic RAG")
    st.caption("Retrieval-Augmented Generation Powered by Google Gemini")
    st.divider()

    # 1. API Key Input Section
    st.markdown("### 🔑 1. Gemini API Key")
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Enter your Google Gemini API Key:",
        value=st.session_state.get("gemini_api_key", env_key),
        type="password",
        placeholder="AIzaSy...",
        help="Your API key is never sent to any third party. It is stored only in your active browser session.",
    )

    if api_key_input != st.session_state.get("gemini_api_key", ""):
        st.session_state.gemini_api_key = api_key_input
        st.session_state.api_key_valid = False
        st.session_state.rag_engine = None

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("Verify Key", use_container_width=True):
            if not api_key_input:
                st.warning("Please enter an API key.")
            else:
                with st.spinner("Validating..."):
                    valid, err = validate_gemini_api_key(api_key_input)
                    if valid:
                        st.session_state.api_key_valid = True
                        st.session_state.last_tested_key = api_key_input
                        st.success("Key is valid!")
                    else:
                        st.session_state.api_key_valid = False
                        st.error(f"Error: {err}")

    with col_btn2:
        st.markdown(
            '<a href="https://aistudio.google.com/app/apikey" target="_blank" style="text-decoration: none;">'
            '<button style="width: 100%; height: 38px; border-radius: 6px; border: 1px solid #4F46E5; background-color: transparent; color: #818CF8; cursor: pointer;">Get Key ↗</button>'
            '</a>',
            unsafe_allow_html=True,
        )

    if st.session_state.api_key_valid:
        st.markdown("🟢 **Status:** API Key Active")
    elif api_key_input:
        st.markdown("🟡 **Status:** Key unverified (click Verify or proceed)")
    else:
        st.markdown("🔴 **Status:** Key Required")

    st.divider()

    # 2. Document Management Section
    st.markdown("### 📂 2. Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "md", "docx", "csv"],
        accept_multiple_files=True,
        help="Upload PDF, Plain Text, Markdown, Word, or CSV documents.",
    )

    col_demo, col_clear = st.columns(2)
    with col_demo:
        load_demo = st.button("📚 Load Demo Docs", use_container_width=True, help="Load pre-bundled sample documents on AI and Quantum Computing.")
    with col_clear:
        clear_index = st.button("🗑️ Clear Index", use_container_width=True, help="Wipe indexed documents and vector store.")

    # Chunking Controls
    with st.expander("⚙️ Chunking Settings", expanded=False):
        chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=800, step=50)
        chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=400, value=150, step=25)

    index_btn = st.button("🚀 Process & Index Documents", type="primary", use_container_width=True)

    st.divider()

    # 3. Model & Retrieval Settings
    st.markdown("### 🎛️ 3. RAG Settings")
    llm_model = st.selectbox(
        "Gemini LLM Model",
        options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        help="Select the Gemini model for synthesis.",
    )

    col_rag1, col_rag2 = st.columns(2)
    with col_rag1:
        top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=4, help="Number of retrieved chunks passed to prompt.")
    with col_rag2:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05, help="Lower values produce more factual, grounded output.")

    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=0.8,
        value=0.25,
        step=0.05,
        help="Minimum cosine similarity required for a chunk to be included.",
    )

    with st.expander("📝 System Prompt & Guardrails", expanded=False):
        system_prompt = st.text_area(
            "Custom System Instruction",
            value=DEFAULT_SYSTEM_PROMPT,
            height=180,
            help="Custom guidelines for answering.",
        )

    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------
# Document Ingestion Logic
# ---------------------------------------------------------
if clear_index:
    if st.session_state.rag_engine:
        st.session_state.rag_engine.vector_store.clear()
    st.session_state.indexed_files = []
    st.sidebar.success("Vector store cleared.")
    st.rerun()

if load_demo:
    if not api_key_input:
        st.sidebar.error("Please enter a Gemini API Key first.")
    else:
        engine = get_active_engine(api_key_input, llm_model, system_prompt)
        if engine:
            with st.spinner("Loading and indexing bundled demo documents..."):
                demo_docs = load_sample_documents()
                if demo_docs:
                    res = engine.ingest_documents(
                        demo_docs,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    st.session_state.indexed_files = sorted(list({d.metadata.get("source", "sample") for d in demo_docs}))
                    st.sidebar.success(f"Indexed {res['chunks']} chunks from demo documents!")
                    st.rerun()
                else:
                    st.sidebar.warning("No sample documents found in data/sample_documents.")

if index_btn:
    if not api_key_input:
        st.sidebar.error("Please enter a Gemini API Key before indexing.")
    elif not uploaded_files:
        st.sidebar.warning("Please upload one or more documents, or use 'Load Demo Docs'.")
    else:
        engine = get_active_engine(api_key_input, llm_model, system_prompt)
        if engine:
            progress_bar = st.sidebar.progress(0.0)
            status_text = st.sidebar.empty()

            def update_progress(ratio: float, msg: str):
                progress_bar.progress(ratio)
                status_text.caption(msg)

            all_docs: list[Document] = []
            file_names = []
            for file in uploaded_files:
                file_names.append(file.name)
                try:
                    file_docs = DocumentLoader.load_from_stream(file, file.name)
                    all_docs.extend(file_docs)
                except Exception as e:
                    st.sidebar.error(f"Error loading {file.name}: {e}")

            if all_docs:
                try:
                    res = engine.ingest_documents(
                        all_docs,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        progress_callback=update_progress,
                    )
                    st.session_state.indexed_files = list(set(st.session_state.indexed_files + file_names))
                    status_text.caption("Complete!")
                    st.sidebar.success(f"Indexed {res['chunks']} chunks across {len(file_names)} files!")
                except Exception as e:
                    st.sidebar.error(f"Indexing error: {e}")
            else:
                st.sidebar.error("Could not extract any readable text from the uploaded files.")


# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.title("⚡ Classic RAG with Google Gemini")
st.markdown(
    "A clean, robust implementation of **Classic Retrieval-Augmented Generation (RAG)**. "
    "Upload your documents, generate embeddings via `text-embedding-004`, retrieve semantically relevant context via cosine similarity, "
    "and synthesize answers using **Google Gemini** with precise source citations."
)

tab_chat, tab_explorer, tab_architecture = st.tabs(["💬 RAG Chat", "🔍 Vector Store Explorer", "ℹ️ Architecture & How It Works"])

# ---------------------------------------------------------
# Tab 1: RAG Chat
# ---------------------------------------------------------
with tab_chat:
    engine = st.session_state.rag_engine
    stats = engine.vector_store.get_stats() if engine else {"total_chunks": 0, "total_documents": 0}

    # Status Pill Metric Bar
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Indexed Documents", stats.get("total_documents", 0))
    with col_m2:
        st.metric("Total Chunks", stats.get("total_chunks", 0))
    with col_m3:
        st.metric("Active LLM", llm_model)
    with col_m4:
        st.metric("Embedding Model", "text-embedding-004")

    st.markdown("---")

    # If no documents are indexed, show helpful guide
    if stats.get("total_chunks", 0) == 0:
        st.info(
            "👋 **Welcome! Get started in 3 simple steps:**\n\n"
            "1. **Enter your Gemini API Key** in the sidebar (or get one from [Google AI Studio](https://aistudio.google.com/app/apikey)).\n"
            "2. **Upload your documents** (PDF, TXT, DOCX, CSV) or click **'📚 Load Demo Docs'**.\n"
            "3. **Click '🚀 Process & Index Documents'** and start asking questions!"
        )

        # Quick starter prompt suggestions
        st.markdown("#### 💡 Example Starter Questions (using bundled demo documents):")
        example_cols = st.columns(3)
        sample_queries = [
            "What was the significance of AlexNet in 2012?",
            "Explain quantum superposition and entanglement.",
            "What factors caused the historical AI Winters?",
        ]
        for col, query_text in zip(example_cols, sample_queries):
            with col:
                if st.button(query_text, use_container_width=True):
                    if not api_key_input:
                        st.error("Please provide a Gemini API Key first.")
                    else:
                        st.session_state.pending_query = query_text
                        st.rerun()

    # Display Chat History
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        with st.chat_message(role):
            st.markdown(content)
            if sources:
                with st.expander(f"📑 View Retrieved Sources ({len(sources)} chunks)", expanded=False):
                    for i, s in enumerate(sources, start=1):
                        page_str = f" • Page {s.get('page')}" if s.get("page") else ""
                        st.markdown(
                            f"**Source #{i}: `{s.get('source')}`**{page_str} "
                            f"— *Relevance Score: `{s.get('similarity_score', 0):.1%}`*"
                        )
                        st.caption(s.get("text", ""))
                        st.divider()

    # Check for pending query from starter button
    user_prompt = st.chat_input("Ask a question about your indexed documents...")
    if not user_prompt and "pending_query" in st.session_state:
        user_prompt = st.session_state.pop("pending_query")

    if user_prompt:
        if not api_key_input:
            st.error("Please enter a Gemini API Key in the sidebar to proceed.")
        else:
            engine = get_active_engine(api_key_input, llm_model, system_prompt)

            if engine is None or engine.vector_store.is_empty():
                st.warning("Your knowledge base is empty. Please upload documents or click 'Load Demo Docs' and index them first.")
            else:
                # Append user message
                st.session_state.messages.append({"role": "user", "content": user_prompt})
                with st.chat_message("user"):
                    st.markdown(user_prompt)

                # Generate Answer with Streaming
                with st.chat_message("assistant"):
                    try:
                        stream_gen, retrieved_sources = engine.generate_answer_stream(
                            query=user_prompt,
                            top_k=top_k,
                            score_threshold=similarity_threshold,
                            temperature=temperature,
                            chat_history=st.session_state.messages[:-1],
                        )

                        full_response = st.write_stream(stream_gen)

                        # Render retrieved context accordion
                        serializable_sources = []
                        if retrieved_sources:
                            with st.expander(f"📑 View Retrieved Sources ({len(retrieved_sources)} chunks)", expanded=False):
                                for i, s in enumerate(retrieved_sources, start=1):
                                    page_str = f" • Page {s.page}" if s.page else ""
                                    st.markdown(
                                        f"**Source #{i}: `{s.source}`**{page_str} "
                                        f"— *Relevance Score: `{s.similarity_score:.1%}`*"
                                    )
                                    st.caption(s.text)
                                    st.divider()

                                    serializable_sources.append({
                                        "source": s.source,
                                        "page": s.page,
                                        "chunk_index": s.chunk_index,
                                        "similarity_score": s.similarity_score,
                                        "text": s.text,
                                    })
                        else:
                            st.caption("ℹ️ No chunks met the similarity threshold.")

                        # Append assistant response to state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "sources": serializable_sources,
                        })

                    except Exception as e:
                        st.error(f"Generation error: {e}")


# ---------------------------------------------------------
# Tab 2: Vector Store Explorer
# ---------------------------------------------------------
with tab_explorer:
    st.subheader("🔍 Vector Store & Semantic Search Inspector")
    st.markdown(
        "Directly test semantic similarity search against the in-memory vector database. "
        "Inspect which chunks match your query and their exact cosine similarity scores."
    )

    engine = st.session_state.rag_engine
    if engine is None or engine.vector_store.is_empty():
        st.info("No documents are currently indexed. Upload files or load demo documents to explore the vector space.")
    else:
        test_col1, test_col2 = st.columns([3, 1])
        with test_col1:
            search_query = st.text_input("Enter search query to inspect retrieval:", placeholder="e.g. quantum entanglement")
        with test_col2:
            test_top_k = st.number_input("Inspect Top-K", min_value=1, max_value=20, value=5)

        if search_query:
            if not api_key_input:
                st.error("Please enter a Gemini API Key to embed the search query.")
            else:
                with st.spinner("Computing query embedding and searching vector store..."):
                    try:
                        matches = engine.retrieve(search_query, top_k=test_top_k, score_threshold=0.0)
                        if matches:
                            st.success(f"Found {len(matches)} matching chunks:")
                            for idx, m in enumerate(matches, start=1):
                                with st.container(border=True):
                                    st.markdown(
                                        f"**Rank #{idx} | Document: `{m.source}` | Page: `{m.page or 'N/A'}` | Chunk Index: `{m.chunk_index}`**"
                                    )
                                    score_pct = m.similarity_score * 100
                                    st.progress(max(0.0, min(1.0, m.similarity_score)), text=f"Cosine Similarity: {score_pct:.2f}%")
                                    st.text_area(f"Chunk Text #{idx}", m.text, height=120, disabled=True)
                        else:
                            st.warning("No matches found.")
                    except Exception as e:
                        st.error(f"Search failed: {e}")

        st.markdown("---")
        st.markdown("### 📚 All Indexed Chunks")
        with st.expander(f"View all {len(engine.vector_store.chunks)} chunks in memory"):
            for i, chunk in enumerate(engine.vector_store.chunks):
                st.markdown(f"**Chunk #{i + 1}** — Source: `{chunk.source}` (Length: {len(chunk.text)} chars)")
                st.caption(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))
                st.divider()


# ---------------------------------------------------------
# Tab 3: Architecture & Documentation
# ---------------------------------------------------------
with tab_architecture:
    st.subheader("🏛️ Classic RAG Architecture")
    st.markdown(r"""
A **Classic RAG (Retrieval-Augmented Generation)** system consists of two distinct phases:

### Phase 1: Ingestion & Indexing (Offline / Setup)
1. **Document Loading**: Multiple file formats (PDF, TXT, Markdown, Word DOCX, CSV) are parsed into standardized `Document` objects.
2. **Text Chunking**: Documents are split into overlapping segments using a `RecursiveCharacterTextSplitter`. Overlap ensures that contextual meaning spanning across boundaries is preserved.
3. **Embedding Generation**: Text chunks are passed through Google's `text-embedding-004` model to produce dense floating-point vector representations (768 dimensions).
4. **Vector Storage**: Vectors are L2-normalized and indexed in an in-memory NumPy matrix for rapid dot-product cosine similarity calculation.

```
[ Documents (PDF, TXT, DOCX) ]
             │
             ▼
   [ Text Splitter (Chunk + Overlap) ]
             │
             ▼
[ Gemini text-embedding-004 ] ──► [ Vector Store (Normalized Matrix) ]
```

### Phase 2: Retrieval & Generation (Online / Query Time)
1. **User Query**: The user asks a question via the chat interface.
2. **Query Embedding**: The query is converted into a vector using the same `text-embedding-004` model.
3. **Semantic Search**: Exact cosine similarity is computed across all indexed chunks ($u \cdot v^T$). The top-$k$ chunks above the similarity threshold are retrieved.
4. **Prompt Augmentation**: The retrieved context chunks, system instructions, and conversation history are synthesized into a structured prompt.
5. **Grounded Generation**: **Google Gemini (e.g. `gemini-2.5-flash`)** streams the final response strictly grounded in the retrieved sources, providing transparency and citations.

```
[ User Query ]
       │
       ▼
[ Query Embedding ] ──► [ Cosine Similarity Search ] ──► [ Top-K Context Chunks ]
                                                                     │
                                                                     ▼
                                                   [ Augmented Grounded Prompt ]
                                                                     │
                                                                     ▼
                                                   [ Google Gemini LLM Response ]
```
""")
