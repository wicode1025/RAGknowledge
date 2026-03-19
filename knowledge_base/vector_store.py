"""
Vector Store using FAISS and Sentence Transformers
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss

from config import Config
from knowledge_base.text_splitter import TextChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store for storing and retrieving text embeddings using FAISS.
    """

    def __init__(
        self,
        embedding_model: str = None,
        embedding_dim: int = None,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None
    ):
        """
        Initialize vector store

        Args:
            embedding_model: Name of sentence-transformers model
            embedding_dim: Dimension of embeddings
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load chunk metadata
        """
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        self.embedding_dim = embedding_dim or Config.EMBEDDING_DIM

        self._model = None
        self._index = None
        self._metadata = []  # Store chunk metadata

        self.index_path = index_path or (Config.VECTORS_DIR / "faiss_index.bin")
        self.metadata_path = metadata_path or (Config.VECTORS_DIR / "metadata.pkl")

    @property
    def model(self):
        """Lazy load embedding model"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._model = SentenceTransformer(self.embedding_model_name)

            # Verify embedding dimension
            test_embedding = self._model.encode(["test"])
            actual_dim = test_embedding.shape[1]
            if actual_dim != self.embedding_dim:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                    f"got {actual_dim}. Updating to {actual_dim}."
                )
                self.embedding_dim = actual_dim

        return self._model

    @property
    def index(self):
        """Lazy initialize FAISS index"""
        if self._index is None:
            # Try to load existing index
            if self.index_path.exists():
                try:
                    self.load()
                    logger.info(f"Loaded existing index from {self.index_path}")
                    return self._index
                except Exception as e:
                    logger.warning(f"Could not load existing index: {e}")

            # Create new index
            logger.info(f"Creating new FAISS index with dimension {self.embedding_dim}")
            self._index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity with normalized vectors)

        return self._index

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> List[int]:
        """
        Add texts to the vector store

        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dicts for each text

        Returns:
            List of chunk IDs
        """
        if not texts:
            return []

        logger.info(f"Adding {len(texts)} texts to vector store")

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True  # Important for cosine similarity
        )

        # Get chunk IDs
        start_id = len(self._metadata)
        chunk_ids = list(range(start_id, start_id + len(texts)))

        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))

        # Store metadata
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            metadata.update({
                'text': text,
                'chunk_id': start_id + i
            })
            self._metadata.append(metadata)

        logger.info(f"Added {len(texts)} texts. Total: {len(self._metadata)}")

        return chunk_ids

    def add_chunks(
        self,
        chunks: List[TextChunk],
        source_name: str = None
    ) -> List[int]:
        """
        Add TextChunk objects to the vector store

        Args:
            chunks: List of TextChunk objects
            source_name: Name of the source document

        Returns:
            List of chunk IDs
        """
        texts = [chunk.text for chunk in chunks]
        metadatas = [
            {
                'source': source_name,
                'start_index': chunk.start_index,
                'end_index': chunk.end_index,
                'chunk_index': chunk.chunk_index
            }
            for chunk in chunks
        ]

        return self.add_texts(texts, metadatas)

    def search(
        self,
        query: str,
        k: int = None,
        filter_metadata: Optional[Dict] = None
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Search for similar texts

        Args:
            query: Query string
            k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            Tuple of (texts, scores, metadatas)
        """
        k = k or Config.TOP_K

        if len(self._metadata) == 0:
            logger.warning("Vector store is empty")
            return [], [], []

        # Generate query embedding
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype('float32')

        # Search
        scores, indices = self.index.search(query_embedding, min(k, len(self._metadata)))

        # Extract results
        results_texts = []
        results_scores = []
        results_metadatas = []

        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._metadata):
                metadata = self._metadata[idx].copy()

                # Remove text from metadata to avoid duplication
                text = metadata.pop('text')

                # Apply metadata filter
                if filter_metadata:
                    matches = all(
                        metadata.get(k) == v
                        for k, v in filter_metadata.items()
                    )
                    if not matches:
                        continue

                results_texts.append(text)
                results_scores.append(float(score))
                results_metadatas.append(metadata)

        return results_texts, results_scores, results_metadatas

    def search_with_threshold(
        self,
        query: str,
        k: int = None,
        threshold: float = None
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Search with similarity threshold filter

        Args:
            query: Query string
            k: Number of results to return
            threshold: Minimum similarity score

        Returns:
            Tuple of (texts, scores, metadatas) filtered by threshold
        """
        threshold = threshold or Config.SIMILARITY_THRESHOLD

        texts, scores, metadatas = self.search(query, k)

        # Filter by threshold
        filtered_texts = []
        filtered_scores = []
        filtered_metadatas = []

        for text, score, metadata in zip(texts, scores, metadatas):
            if score >= threshold:
                filtered_texts.append(text)
                filtered_scores.append(score)
                filtered_metadatas.append(metadata)

        return filtered_texts, filtered_scores, filtered_metadatas

    def save(self, index_path: Path = None, metadata_path: Path = None):
        """
        Save index and metadata to disk

        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        index_path = index_path or self.index_path
        metadata_path = metadata_path or self.metadata_path

        # Create directory if needed
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")

        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self._metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")

    def load(self, index_path: Path = None, metadata_path: Path = None):
        """
        Load index and metadata from disk

        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata
        """
        index_path = index_path or self.index_path
        metadata_path = metadata_path or self.metadata_path

        # Load FAISS index
        self._index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index from {index_path}")

        # Load metadata
        with open(metadata_path, 'rb') as f:
            self._metadata = pickle.load(f)
        logger.info(f"Loaded {len(self._metadata)} metadata entries")

    def get_all_texts(self) -> List[str]:
        """Get all stored texts"""
        return [m['text'] for m in self._metadata]

    def __len__(self):
        """Return number of stored texts"""
        return len(self._metadata)

    def is_empty(self) -> bool:
        """Check if vector store is empty"""
        return len(self._metadata) == 0


class InMemoryVectorStore:
    """
    Simple in-memory vector store without FAISS.
    Useful for small-scale testing or when FAISS is not available.
    """

    def __init__(
        self,
        embedding_model: str = None,
        embedding_dim: int = None
    ):
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        self.embedding_dim = embedding_dim or Config.EMBEDDING_DIM
        self._model = None
        self._texts = []
        self._embeddings = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
            test_embedding = self._model.encode(["test"])
            self.embedding_dim = test_embedding.shape[1]
        return self._model

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Add texts to the store"""
        self._texts.extend(texts)

        # Generate embeddings
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

    def search(self, query: str, k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """Search using cosine similarity"""
        if not self._texts:
            return [], [], []

        query_embedding = self.model.encode([query], normalize_embeddings=True)

        # Compute similarities
        similarities = np.dot(self._embeddings, query_embedding.T).flatten()

        # Get top-k
        top_k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        scores = []
        metadatas = []

        for idx in top_indices:
            results.append(self._texts[idx])
            scores.append(float(similarities[idx]))
            metadatas.append({'index': idx})

        return results, scores, metadatas


if __name__ == "__main__":
    # Test vector store
    logging.basicConfig(level=logging.INFO)

    store = VectorStore()

    # Test texts
    test_texts = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing deals with text data.",
        "Computer vision enables machines to understand images.",
    ]

    # Add texts
    store.add_texts(test_texts)
    print(f"Added {len(test_texts)} texts")

    # Save
    store.save()
    print("Saved index")

    # Search
    results, scores, metadatas = store.search("What is deep learning?", k=3)
    print("\nSearch results:")
    for text, score in zip(results, scores):
        print(f"  Score: {score:.4f} | {text[:50]}...")
