# ⚡ Classic RAG with Streamlit & Google Gemini

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-API%20%26%20Embeddings-8E75C2.svg)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, production-grade **Classic Retrieval-Augmented Generation (RAG)** application built with **Streamlit** and the official **Google Gemini GenAI SDK**.

Users can upload documents in various formats (PDF, TXT, DOCX, Markdown, CSV), enter their Gemini API key directly into the Streamlit interface (securely held only in the browser session), index content into an in-memory cosine-similarity vector store, and chat with Google Gemini models using grounded answers with complete source citations.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Obtaining a Gemini API Key](#obtaining-a-gemini-api-key)
  - [Running the Application](#running-the-application)
- [How to Use](#how-to-use)
- [RAG Parameters Guide](#rag-parameters-guide)
- [Running Unit Tests](#running-unit-tests)
- [License](#license)

---

## 🎯 Overview

Retrieval-Augmented Generation (RAG) overcomes the limitation of LLMs' static parametric knowledge by grounding responses in retrieved, up-to-date document chunks.

This project implements the canonical ("Classic") RAG pipeline:
1. **Document Loading**: Extracting raw text and page metadata from documents.
2. **Text Chunking**: Recursive character splitting with sliding-window overlap.
3. **Dense Embeddings**: Converting text chunks into semantic vector spaces via `text-embedding-004`.
4. **Vector Storage & Search**: Computing exact cosine similarity between query and chunk vectors.
5. **Prompt Augmentation & Grounding**: Synthesizing the retrieved context with strict grounding guardrails.
6. **Streaming Generation**: Streaming answers from Gemini models with source attribution and confidence metrics.

---

## 🏛️ Architecture

```
                    ┌────────────────────────────┐
                    │ Document Ingestion Pipeline│
                    └─────────────┬──────────────┘
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    ▼                             ▼                             ▼
[.pdf Documents]            [.txt / .md Files]          [.docx / .csv Data]
    │                             │                             │
    └─────────────────────────────┼─────────────────────────────┘
                                  ▼
                    ┌────────────────────────────┐
                    │  DocumentLoader (Metadata) │
                    └─────────────┬──────────────┘
                                  ▼
                    ┌────────────────────────────┐
                    │ Recursive Character Split  │
                    │  (Chunk Size + Overlap)    │
                    └─────────────┬──────────────┘
                                  ▼
                    ┌────────────────────────────┐
                    │ Google Gemini Embeddings   │
                    │   (text-embedding-004)     │
                    └─────────────┬──────────────┘
                                  ▼
                    ┌────────────────────────────┐
                    │   In-Memory Vector Store   │
                    │  (L2 Normalized Vectors)   │
                    └─────────────┬──────────────┘
                                  │
══════════════════════════════════╪══════════════════════════════════════
                                  │ (Query Time Retrieval)
                                  ▼
[User Query] ──► [Query Embedding] ──► [Cosine Similarity (u · v)]
                                                    │
                                                    ▼
                                         [Top-K Context Chunks]
                                                    │
                                                    ▼
                                         [Augmented Context Prompt]
                                                    │
                                                    ▼
                                         [Google Gemini Generation]
                                                    │
                                                    ▼
                                         [Streamed Answer + Citations]
```

---

## 📁 Project Structure

```
RAG/
├── .streamlit/
│   └── config.toml                  # Custom dark theme and server settings
├── data/
│   └── sample_documents/            # Bundled demo documents for instant testing
│       ├── artificial_intelligence_history.txt
│       └── quantum_computing_primer.md
├── src/
│   ├── __init__.py                  # Package initializer
│   ├── document_loader.py           # Multi-format document parser (PDF, TXT, MD, DOCX, CSV)
│   ├── text_splitter.py             # Recursive character text chunker with overlap
│   ├── embeddings.py                # Gemini embedding client with batching and error handling
│   ├── vector_store.py              # Zero-dependency NumPy cosine similarity index
│   ├── rag_engine.py                # Orchestrator for retrieval, context assembly, and generation
│   └── utils.py                     # API key validator, token estimation, and helpers
├── tests/
│   └── test_rag.py                  # Unit and integration test suite
├── .env.example                     # Environment template (optional)
├── .gitignore                       # Git ignore rules
├── app.py                           # Interactive Streamlit application
├── requirements.txt                 # Pinned dependencies
└── README.md                        # Documentation
```

---

## ✨ Key Features

- **User-Provided API Key**: No hardcoded API keys. Users enter their Google Gemini key in the Streamlit UI with password masking, verification testing, and session isolation.
- **Multiple File Format Support**:
  - Adobe PDF (`.pdf`) with page tracking
  - Plain Text (`.txt`) and Markdown (`.md`)
  - Microsoft Word (`.docx`)
  - Comma-Separated Values (`.csv`) formatted as structured records
- **Zero-Dependency Vector Store**: Powered by NumPy vector operations with exact cosine similarity matching, requiring no heavy external database or complex C++ dependencies.
- **1-Click Demo Data**: Includes built-in sample documents covering AI History and Quantum Computing for immediate out-of-the-box exploration without needing external files.
- **Streaming LLM Responses**: Real-time token streaming with Gemini models (`gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-flash`, `gemini-1.5-pro`).
- **Transparent Source Citations**: Expandable source viewer under every answer showing the exact retrieved chunks, document name, page number, and similarity score percentage.
- **Vector Store Explorer**: A dedicated tab to inspect raw similarity search results and test retrieval mechanics independently from LLM generation.
- **Configurable Hyperparameters**: Easily adjust Chunk Size, Chunk Overlap, Top-K retrieved chunks, Similarity Threshold, and Generation Temperature from the UI.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- A free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sohamhazraff-beep/RAG.git
   cd RAG
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Obtaining a Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Click **"Create API key"**.
4. Copy the key and paste it into the Streamlit app sidebar.
*(Optional)* You can also set it in a `.env` file as `GEMINI_API_KEY=your_key_here` to prefill the field automatically.

### Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

---

## 💡 How to Use

1. **Enter Your Gemini API Key**: In the sidebar, paste your API key and click **"Verify Key"**.
2. **Add Documents to Knowledge Base**:
   - Drag and drop your files (`.pdf`, `.txt`, `.docx`, etc.) into the file uploader, or
   - Click **"📚 Load Demo Docs"** to index the bundled sample documents.
3. **Index Knowledge Base**: Click **"🚀 Process & Index Documents"**. The app will extract text, chunk it, generate embeddings, and build the vector index.
4. **Chat**: Ask questions in the chat box! Gemini will answer strictly grounded in your documents and show expandable source citations.
5. **Inspect the Vector Store**: Switch to the **"🔍 Vector Store Explorer"** tab to test raw similarity search on specific queries.

---

## 🎛️ RAG Parameters Guide

| Parameter | Recommended | Description |
|---|---|---|
| **Chunk Size** | `600 - 1000` | Number of characters per segment. Smaller chunks yield higher precision; larger chunks provide broader context. |
| **Chunk Overlap** | `100 - 200` | Number of overlapping characters between adjacent chunks to maintain context across split boundaries. |
| **Top-K Chunks** | `3 - 5` | Number of top-ranked matching chunks injected into the prompt context for Gemini. |
| **Similarity Threshold** | `0.20 - 0.35` | Minimum cosine similarity required for a chunk to be retrieved. Filters out irrelevant chunks. |
| **Temperature** | `0.1 - 0.3` | Controls model randomness. Keep low for factual RAG to reduce hallucination risk. |

---

## 🧪 Running Unit Tests

The project includes unit tests for the document loader, text splitter, vector store, and utility functions:

```bash
python3 -m unittest discover tests -v
```

All tests run locally in milliseconds without requiring network calls or API keys.

---

## 📜 License

This project is open-source and licensed under the [MIT License](LICENSE).
