"""Unit and Integration Tests for the Classic RAG Project."""

import io
import unittest
import numpy as np
from src.document_loader import Document, DocumentLoader
from src.text_splitter import RecursiveCharacterTextSplitter, TextChunk
from src.vector_store import VectorStore
from src.utils import estimate_tokens, format_file_size, load_sample_documents


class TestDocumentLoader(unittest.TestCase):
    """Test loading documents from text streams and files."""

    def test_load_text_stream(self):
        content = b"This is a test plain text document for RAG ingestion."
        stream = io.BytesIO(content)
        docs = DocumentLoader.load_from_stream(stream, "sample.txt")

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["source"], "sample.txt")
        self.assertEqual(docs[0].metadata["file_type"], "txt")
        self.assertIn("plain text document", docs[0].content)

    def test_load_csv_stream(self):
        csv_data = b"title,author,year\nDeep Learning,Goodfellow,2016\nAttention,Vaswani,2017\n"
        stream = io.BytesIO(csv_data)
        docs = DocumentLoader.load_from_stream(stream, "papers.csv")

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["file_type"], "csv")
        self.assertEqual(docs[0].metadata["num_rows"], 2)
        self.assertIn("Deep Learning", docs[0].content)
        self.assertIn("Attention", docs[0].content)

    def test_load_sample_documents(self):
        samples = load_sample_documents()
        self.assertGreater(len(samples), 0)
        sources = {d.metadata.get("source") for d in samples}
        self.assertTrue(any("artificial_intelligence" in s for s in sources))


class TestTextSplitter(unittest.TestCase):
    """Test recursive character text splitting."""

    def setUp(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

    def test_chunking_and_overlap(self):
        sample_text = (
            "Artificial intelligence is transforming industries across the globe. "
            "From healthcare to finance, machine learning models enable rapid data analysis. "
            "Natural language processing allows computers to interpret human language. "
            "Computer vision allows machines to understand visual scenes."
        )
        chunks = self.splitter.split_text(sample_text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 150)  # reasonable boundary margin

    def test_split_documents_metadata(self):
        doc = Document(
            content="Alpha section text.\n\nBeta section text.\n\nGamma section text.",
            metadata={"source": "test.txt", "author": "Tester"},
        )
        chunks = self.splitter.split_documents([doc])
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["source"], "test.txt")
        self.assertEqual(chunks[0].metadata["author"], "Tester")
        self.assertIn("chunk_index", chunks[0].metadata)


class TestVectorStore(unittest.TestCase):
    """Test vector storage, normalization, and cosine similarity retrieval."""

    def setUp(self):
        self.vs = VectorStore()
        self.chunks = [
            TextChunk(text="Artificial intelligence and neural networks", metadata={"source": "ai.txt"}),
            TextChunk(text="Quantum computing and superposition", metadata={"source": "quantum.txt"}),
            TextChunk(text="Solar energy and photovoltaic cells", metadata={"source": "energy.txt"}),
        ]
        # Generate orthogonal dummy embeddings
        self.embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        self.vs.add_chunks(self.chunks, self.embeddings)

    def test_stats(self):
        stats = self.vs.get_stats()
        self.assertEqual(stats["total_chunks"], 3)
        self.assertEqual(stats["total_documents"], 3)
        self.assertEqual(stats["embedding_dimension"], 3)

    def test_similarity_search_exact_match(self):
        query_vec = np.array([1.0, 0.1, 0.0], dtype=np.float32)
        results = self.vs.similarity_search(query_vec, top_k=2)

        self.assertEqual(len(results), 2)
        top_chunk, score = results[0]
        self.assertEqual(top_chunk.source, "ai.txt")
        self.assertGreater(score, 0.95)

    def test_similarity_threshold(self):
        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = self.vs.similarity_search(query_vec, top_k=3, score_threshold=0.5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].source, "ai.txt")

    def test_clear(self):
        self.vs.clear()
        self.assertTrue(self.vs.is_empty())
        self.assertEqual(self.vs.get_stats()["total_chunks"], 0)


class TestUtilities(unittest.TestCase):
    """Test helper functions."""

    def test_estimate_tokens(self):
        text = "Hello world! This is a test."
        tokens = estimate_tokens(text)
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, len(text))

    def test_format_file_size(self):
        self.assertEqual(format_file_size(500), "500.0 B")
        self.assertEqual(format_file_size(2048), "2.0 KB")
        self.assertEqual(format_file_size(1048576 * 5), "5.0 MB")


if __name__ == "__main__":
    unittest.main()
