"""
RAG Retriever using FAISS for vector similarity search.
Stores JD embeddings and retrieves relevant context for question generation.
"""
import os
import json
import pickle
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import faiss
from loguru import logger

from rag.chunking import chunk_text, smart_chunk_jd, Chunk
from rag.embeddings import SentenceTransformerEmbedder, get_embedder
from config import settings


class FAISSRetriever:
    """
    FAISS-based document retriever for RAG.
    Handles indexing and retrieval of JD chunks.
    """

    def __init__(self, index_path: Optional[str] = None):
        self.embedder = get_embedder(use_openai=False)
        self.index: Optional[faiss.IndexFlatIP] = None  # Inner product (cosine after L2 norm)
        self.chunks: List[Chunk] = []
        self.chunk_texts: List[str] = []
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self._initialized = False

    def _init_index(self, dim: int):
        """Initialize FAISS index."""
        self.index = faiss.IndexFlatIP(dim)  # Inner product for normalized vectors = cosine
        logger.info(f"FAISS index initialized with dim={dim}")

    def add_documents(self, text: str, source: str = "job_description"):
        """
        Add a document to the vector store.
        
        Args:
            text: Document text
            source: Source identifier
        """
        logger.info(f"Chunking document: {source}")
        
        if source == "job_description":
            chunks = smart_chunk_jd(text)
        else:
            chunks = chunk_text(text, source=source)

        if not chunks:
            logger.warning("No chunks generated from document")
            return

        chunk_texts = [c.text for c in chunks]

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedder.embed(chunk_texts)

        if self.index is None:
            self._init_index(embeddings.shape[1])

        # Add to FAISS
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)
        self.chunk_texts.extend(chunk_texts)
        self._initialized = True

        logger.info(f"Added {len(chunks)} chunks. Total chunks: {len(self.chunks)}")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of dicts with 'text', 'score', 'source'
        """
        if not self._initialized or self.index is None or len(self.chunks) == 0:
            logger.warning("Retriever not initialized or empty. No context available.")
            return []

        # Embed query
        query_embedding = self.embedder.embed_single(query).reshape(1, -1).astype(np.float32)

        # Search
        scores, indices = self.index.search(query_embedding, min(top_k * 2, len(self.chunks)))

        results = []
        seen_texts = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            if float(score) < score_threshold:
                continue

            chunk = self.chunks[idx]
            text = chunk.text

            # Deduplicate similar chunks
            text_key = text[:100]
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            results.append({
                "text": text,
                "score": float(score),
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
            })

            if len(results) >= top_k:
                break

        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:60]}...'")
        return results

    def retrieve_context_string(self, query: str, top_k: int = 3) -> str:
        """
        Get retrieved context as a formatted string for LLM prompts.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return "No job description context available."

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Relevant JD Section {i}]\n{result['text']}")

        return "\n\n".join(context_parts)

    def save(self, path: Optional[str] = None):
        """Save index and chunks to disk."""
        save_path = Path(path or self.index_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, str(save_path / "index.faiss"))

        # Save chunks
        with open(save_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info(f"Retriever saved to {save_path}")

    def load(self, path: Optional[str] = None) -> bool:
        """Load index and chunks from disk."""
        load_path = Path(path or self.index_path)
        index_file = load_path / "index.faiss"
        chunks_file = load_path / "chunks.pkl"

        if not index_file.exists() or not chunks_file.exists():
            logger.warning(f"No saved index found at {load_path}")
            return False

        try:
            self.index = faiss.read_index(str(index_file))
            with open(chunks_file, "rb") as f:
                self.chunks = pickle.load(f)
            self.chunk_texts = [c.text for c in self.chunks]
            self._initialized = True
            logger.info(f"Loaded {len(self.chunks)} chunks from {load_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load retriever: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if retriever has indexed documents."""
        return self._initialized and len(self.chunks) > 0

    def reset(self):
        """Clear the index."""
        self.index = None
        self.chunks = []
        self.chunk_texts = []
        self._initialized = False
        logger.info("Retriever reset")


# Session-scoped retriever cache
_retrievers: Dict[str, FAISSRetriever] = {}


def get_retriever(session_id: str) -> FAISSRetriever:
    """Get or create a retriever for a session."""
    if session_id not in _retrievers:
        _retrievers[session_id] = FAISSRetriever()
    return _retrievers[session_id]


def clear_retriever(session_id: str):
    """Clear retriever for a session."""
    if session_id in _retrievers:
        del _retrievers[session_id]