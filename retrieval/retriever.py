"""
Retriever Module
Implements retrieval logic for RAG system using FAISS and embeddings.
"""

import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np

from knowledge_base.vector_store import VectorStore
from config import Config

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieval result"""
    text: str
    score: float
    metadata: Dict
    rank: int


class Retriever:
    """
    Retriever for finding relevant context from vector store.
    """

    def __init__(
        self,
        vector_store: VectorStore = None,
        top_k: int = None,
        similarity_threshold: float = None
    ):
        """
        Initialize retriever

        Args:
            vector_store: Vector store instance
            top_k: Number of results to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k or Config.TOP_K
        self.similarity_threshold = similarity_threshold or Config.SIMILARITY_THRESHOLD

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_threshold: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant context for a query

        Args:
            query: User query
            top_k: Number of results to retrieve (overrides default)
            use_threshold: Whether to apply similarity threshold

        Returns:
            List of RetrievalResult objects
        """
        k = top_k or self.top_k

        if self.vector_store.is_empty():
            logger.warning("Vector store is empty, cannot retrieve")
            return []

        logger.info(f"Retrieving top {k} results for query: {query[:50]}...")

        if use_threshold:
            texts, scores, metadatas = self.vector_store.search_with_threshold(
                query,
                k=k,
                threshold=self.similarity_threshold
            )
        else:
            texts, scores, metadatas = self.vector_store.search(query, k=k)

        # Convert to RetrievalResult objects
        results = []
        for i, (text, score, metadata) in enumerate(zip(texts, scores, metadatas)):
            results.append(RetrievalResult(
                text=text,
                score=score,
                metadata=metadata,
                rank=i + 1
            ))

        logger.info(f"Retrieved {len(results)} results above threshold {self.similarity_threshold}")

        return results

    def retrieve_with_fallback(
        self,
        query: str,
        top_k: int = None
    ) -> Tuple[List[RetrievalResult], bool]:
        """
        Retrieve with fallback to all results if threshold not met

        Args:
            query: User query
            top_k: Number of results to retrieve

        Returns:
            Tuple of (results, threshold_met)
            - results: List of RetrievalResult
            - threshold_met: Whether any result met the similarity threshold
        """
        k = top_k or self.top_k

        # First try with threshold
        results = self.retrieve(query, top_k=k, use_threshold=True)

        if results:
            return results, True

        # Fallback: retrieve without threshold
        logger.info("No results above threshold, falling back to top results")
        results = self.retrieve(query, top_k=k, use_threshold=False)

        return results, False

    def format_context(
        self,
        results: List[RetrievalResult],
        include_scores: bool = True,
        include_metadata: bool = False
    ) -> str:
        """
        Format retrieval results as context string

        Args:
            results: List of RetrievalResult objects
            include_scores: Whether to include similarity scores
            include_metadata: Whether to include metadata

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."

        context_parts = ["Context Information:"]

        for i, result in enumerate(results):
            part = f"\n[Source {i + 1}]"
            if include_scores:
                part += f" (Score: {result.score:.4f})"
            part += f"\n{result.text}"
            context_parts.append(part)

            if include_metadata and result.metadata:
                part += f"\n  Metadata: {result.metadata}"

        return "\n".join(context_parts)

    def get_unique_sources(self, results: List[RetrievalResult]) -> List[str]:
        """
        Get unique source names from results

        Args:
            results: List of RetrievalResult objects

        Returns:
            List of unique source names
        """
        sources = set()
        for result in results:
            source = result.metadata.get('source', 'unknown')
            sources.add(source)
        return sorted(sources)


class HybridRetriever(Retriever):
    """
    Hybrid retriever combining multiple retrieval strategies.
    Currently supports: dense (embedding) retrieval.
    """

    def __init__(
        self,
        vector_store: VectorStore = None,
        top_k: int = None,
        similarity_threshold: float = None,
        rerank: bool = False
    ):
        super().__init__(vector_store, top_k, similarity_threshold)
        self.rerank = rerank

    def retrieve_dense(
        self,
        query: str,
        top_k: int = None
    ) -> List[RetrievalResult]:
        """Dense retrieval using embeddings"""
        return self.retrieve(query, top_k)

    def retrieve_with_diversity(
        self,
        query: str,
        top_k: int = None,
        diversity_weight: float = 0.3
    ) -> List[RetrievalResult]:
        """
        Retrieve with diversity promotion (MMR-like)

        Args:
            query: User query
            top_k: Number of results
            diversity_weight: Weight for diversity (0-1)

        Returns:
            List of diverse RetrievalResult objects
        """
        k = top_k or self.top_k
        k_diverse = max(k * 2, 10)  # Get more candidates

        # Get initial results
        results = self.retrieve(query, top_k=k_diverse, use_threshold=False)

        if len(results) <= k:
            return results

        # Select diverse results
        selected = [results[0]]
        remaining = results[1:]

        while len(selected) < k and remaining:
            best_idx = -1
            best_score = -float('inf')

            for i, result in enumerate(remaining):
                # Relevance score
                relevance = result.score

                # Diversity score (min similarity to selected)
                diversity = 1.0
                for sel in selected:
                    # Compute text similarity (simple word overlap)
                    sel_words = set(sel.text.lower().split())
                    result_words = set(result.text.lower().split())
                    if sel_words and result_words:
                        overlap = len(sel_words & result_words) / len(sel_words | result_words)
                        diversity = min(diversity, 1 - overlap)

                # Combined score
                combined = (1 - diversity_weight) * relevance + diversity_weight * diversity

                if combined > best_score:
                    best_score = combined
                    best_idx = i

            if best_idx >= 0:
                selected.append(remaining[best_idx])
                remaining.pop(best_idx)
            else:
                break

        return selected


class BM25Retriever:
    """
    BM25 sparse retriever (optional addition for hybrid search).
    Note: Requires rank_bm25 package
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = {}
        self.idf = {}
        self.corpus = []

    def add_documents(self, documents: List[str]):
        """Add documents to the index"""
        self.corpus = documents
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if documents else 0

        # Calculate document frequencies
        for doc in documents:
            words = set(doc.lower().split())
            for word in words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        # Calculate IDF
        N = len(documents)
        for word, df in self.doc_freqs.items():
            self.idf[word] = np.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_scores(self, query: str) -> np.ndarray:
        """Get BM25 scores for a query"""
        query_words = query.lower().split()
        scores = np.zeros(len(self.corpus))

        for doc_idx, doc in enumerate(self.corpus):
            doc_words = doc.lower().split()
            doc_len = self.doc_lengths[doc_idx]
            word_freq = {}

            for word in doc_words:
                word_freq[word] = word_freq.get(word, 0) + 1

            for word in query_words:
                if word in word_freq:
                    tf = word_freq[word]
                    idf = self.idf.get(word, 0)

                    # BM25 scoring formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    scores[doc_idx] += idf * numerator / denominator

        return scores

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Retrieve top-k documents"""
        scores = self.get_scores(query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]


if __name__ == "__main__":
    # Test retriever
    logging.basicConfig(level=logging.INFO)

    # Create vector store and add test data
    store = VectorStore()
    test_texts = [
        "Python is a high-level programming language known for its readability.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn.",
        "Deep learning uses neural networks with multiple layers to learn representations.",
        "Natural language processing deals with understanding and generating human language.",
        "Computer vision enables machines to interpret and understand visual information.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Transformers are a type of architecture used in natural language processing.",
    ]

    store.add_texts(test_texts)

    # Create retriever
    retriever = Retriever(vector_store=store, top_k=3, similarity_threshold=0.5)

    # Test retrieval
    query = "What is deep learning?"
    results = retriever.retrieve(query)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} results:")
    for result in results:
        print(f"  Rank {result.rank}: Score={result.score:.4f}")
        print(f"    Text: {result.text[:80]}...")
