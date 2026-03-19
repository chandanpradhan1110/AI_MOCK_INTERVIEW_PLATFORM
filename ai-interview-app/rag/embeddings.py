"""
Embeddings module using SentenceTransformers for local, free embeddings.
Falls back to OpenAI embeddings if configured.
"""
from typing import List, Optional
import numpy as np
from loguru import logger


class SentenceTransformerEmbedder:
    """
    Local embeddings using SentenceTransformers.
    No API costs. Works offline.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        logger.info(f"SentenceTransformerEmbedder initialized with model: {model_name}")

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        self._load_model()
        if not texts:
            return np.array([])

        embeddings = self._model.encode(
            texts,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()


class OpenAIEmbedder:
    """
    OpenAI embeddings (higher quality but costs money).
    Use when SentenceTransformers quality is insufficient.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        from config import settings
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model
        self._dim = 1536 if "small" in model else 3072
        logger.info(f"OpenAIEmbedder initialized with model: {model}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using OpenAI API."""
        if not texts:
            return np.array([])

        # Batch in groups of 100
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        arr = np.array(all_embeddings, dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norms, 1e-8)
        return arr

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dim


def get_embedder(use_openai: bool = False):
    """
    Factory function to get the appropriate embedder.
    
    Args:
        use_openai: If True, use OpenAI embeddings
        
    Returns:
        Embedder instance
    """
    if use_openai:
        try:
            return OpenAIEmbedder()
        except Exception as e:
            logger.warning(f"OpenAI embedder failed, falling back to local: {e}")

    return SentenceTransformerEmbedder()