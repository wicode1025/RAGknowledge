"""
RAG System
Main system that combines knowledge base, retrieval, evaluation, and LLM.
"""

import os
import logging
import pickle
import faiss
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from config import Config
from knowledge_base.nougat_parser import NougatParser, SimpleMarkdownParser
from knowledge_base.text_splitter import RecursiveCharacterTextSplitter, MarkdownAwareSplitter
from knowledge_base.vector_store import VectorStore
from retrieval.retriever import Retriever, HybridRetriever
from evaluation.threshold_filter import ThresholdFilter, EvaluationResult, RetrievalStatus
from llm.claude_client import ClaudeClient, GenerationResult
from llm.qwen_client import QwenClient

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG system"""
    answer: str
    success: bool
    status: RetrievalStatus
    message: str
    sources: List[Dict]
    max_similarity: float
    evaluation: Optional[EvaluationResult] = None


class RAGSystem:
    """
    Main RAG system combining all components.
    """

    def __init__(
        self,
        api_key: str = None,
        embedding_model: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        top_k: int = None,
        similarity_threshold: float = None,
        use_markdown_splitter: bool = True,
        use_nougat: bool = False,
        llm_provider: str = None
    ):
        """
        Initialize RAG system

        Args:
            api_key: API key for LLM (Claude or Qwen)
            embedding_model: Embedding model name
            chunk_size: Chunk size for text splitting
            chunk_overlap: Chunk overlap
            top_k: Number of context chunks to retrieve
            similarity_threshold: Minimum similarity threshold
            use_markdown_splitter: Whether to use Markdown-aware splitter
            use_nougat: Whether to use Nougat for PDF parsing
            llm_provider: LLM provider - "qwen" or "claude"
        """
        # Configuration
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.top_k = top_k or Config.TOP_K
        self.similarity_threshold = similarity_threshold or Config.SIMILARITY_THRESHOLD

        # LLM provider
        self.llm_provider = llm_provider or Config.LLM_PROVIDER
        self._api_key = api_key

        # Initialize components
        self.vector_store = VectorStore(embedding_model=self.embedding_model)
        self.retriever = Retriever(
            vector_store=self.vector_store,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold
        )
        self.threshold_filter = ThresholdFilter(
            threshold=self.similarity_threshold,
            min_results=1
        )
        self.llm_client = None  # Lazy initialization
        self._api_key = api_key

        # Text splitter
        if use_markdown_splitter:
            self.text_splitter = MarkdownAwareSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

        # PDF parser
        self.use_nougat = use_nougat
        if use_nougat:
            try:
                self.pdf_parser = NougatParser()
            except Exception as e:
                logger.warning(f"Failed to load Nougat: {e}. Using simple parser.")
                self.pdf_parser = SimpleMarkdownParser()
        else:
            self.pdf_parser = SimpleMarkdownParser()

        self._initialized = False

    @property
    def llm(self):
        """Lazy initialize LLM client based on provider"""
        if self.llm_client is None:
            if self.llm_provider == "qwen":
                self.llm_client = QwenClient(api_key=self._api_key)
            else:
                self.llm_client = ClaudeClient(api_key=self._api_key)
        return self.llm_client

    def initialize(self, pdf_dir: Path = None, markdown_dir: Path = None):
        """
        Initialize the RAG system by loading or building knowledge base

        Args:
            pdf_dir: Directory containing PDF files
            markdown_dir: Directory containing pre-processed markdown files
        """
        # Try to load existing index
        import pickle
        import faiss

        vectors_dir = Config.VECTORS_DIR
        index_path = vectors_dir / "faiss_index.bin"
        metadata_path = vectors_dir / "metadata.pkl"

        if index_path.exists() and metadata_path.exists():
            try:
                # Load index directly to ensure consistency
                self.vector_store._index = faiss.read_index(str(index_path))
                with open(metadata_path, 'rb') as f:
                    self.vector_store._metadata = pickle.load(f)
                logger.info(f"Loaded existing vector index with {len(self.vector_store)} chunks")

                # Recreate retriever with loaded vector store
                self.retriever = Retriever(
                    vector_store=self.vector_store,
                    top_k=self.top_k,
                    similarity_threshold=self.similarity_threshold
                )

                self._initialized = True
                return
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")

        # Build new knowledge base
        self.build_knowledge_base(pdf_dir=pdf_dir, markdown_dir=markdown_dir)
        self._initialized = True

    def build_knowledge_base(
        self,
        pdf_dir: Path = None,
        markdown_dir: Path = None,
        force_rebuild: bool = False
    ):
        """
        Build knowledge base from documents

        Args:
            pdf_dir: Directory containing PDF files
            markdown_dir: Directory containing markdown files
            force_rebuild: Whether to rebuild even if index exists
        """
        pdf_dir = pdf_dir or Config.PDFS_DIR
        markdown_dir = markdown_dir or Config.PROCESSED_DIR

        logger.info("Building knowledge base...")

        all_texts = []

        # Process markdown files
        if markdown_dir.exists():
            logger.info(f"Processing markdown files from {markdown_dir}")
            for md_file in markdown_dir.glob("*.md"):
                logger.info(f"Loading: {md_file.name}")
                with open(md_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    all_texts.append((text, md_file.stem))

        # Process PDF files (including subdirectories)
        if pdf_dir.exists():
            logger.info(f"Processing PDF files from {pdf_dir}")
            # Use rglob to find PDFs in subdirectories too
            pdf_files = list(pdf_dir.rglob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} PDF files")

            for pdf_file in pdf_files:
                logger.info(f"Parsing: {pdf_file.name}")
                try:
                    text = self.pdf_parser.parse_pdf(
                        pdf_file,
                        output_dir=markdown_dir
                    )
                    all_texts.append((text, pdf_file.stem))
                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {e}")

        if not all_texts:
            logger.warning("No documents found to process")
            return

        # Split texts into chunks
        all_chunks = []
        all_metadatas = []

        for text, source_name in all_texts:
            logger.info(f"Splitting {source_name} into chunks...")
            chunks = self.text_splitter.create_chunks(text, source_name=source_name)

            for chunk in chunks:
                all_chunks.append(chunk.text)
                all_metadatas.append({
                    'source': source_name,
                    'chunk_index': chunk.chunk_index
                })

        logger.info(f"Created {len(all_chunks)} chunks from {len(all_texts)} documents")

        # Add to vector store
        self.vector_store.add_texts(all_chunks, metadatas=all_metadatas)

        # Save index
        self.vector_store.save()

        logger.info(f"Knowledge base built with {len(self.vector_store)} chunks")

    def query(
        self,
        question: str,
        use_llm: bool = True,
        return_sources: bool = True,
        return_context: bool = False
    ) -> RAGResponse:
        """
        Query the RAG system

        Args:
            question: User question
            use_llm: Whether to use LLM for generation
            return_sources: Whether to include source information
            return_context: Whether to include retrieved context

        Returns:
            RAGResponse object
        """
        if not self._initialized:
            self.initialize()

        # Step 1: Retrieve context
        logger.info(f"Processing query: {question[:50]}...")
        results = self.retriever.retrieve(question)

        # Step 2: Evaluate with threshold filter
        evaluation = self.threshold_filter.evaluate(question, results)

        # Step 3: Handle based on evaluation result
        if not evaluation.can_proceed:
            # Return rejection response
            return RAGResponse(
                answer=self.threshold_filter.get_rejection_response(
                    question,
                    evaluation.max_score
                ),
                success=False,
                status=evaluation.status,
                message=evaluation.message,
                sources=[],
                max_similarity=evaluation.max_score,
                evaluation=evaluation
            )

        # Step 4: Format context
        context = self.retriever.format_context(evaluation.results)

        # Step 5: Generate LLM response if requested
        if use_llm:
            llm_result = self.llm.generate_with_sources(
                prompt=question,
                context=context
            )
            answer = llm_result.text
        else:
            # Just return context as answer
            answer = context

        # Step 6: Format sources
        sources = []
        if return_sources:
            for result in evaluation.results:
                sources.append({
                    'text': result.text[:200] + '...' if len(result.text) > 200 else result.text,
                    'score': result.score,
                    'source': result.metadata.get('source', 'unknown')
                })

        return RAGResponse(
            answer=answer,
            success=True,
            status=evaluation.status,
            message=evaluation.message,
            sources=sources,
            max_similarity=evaluation.max_score,
            evaluation=evaluation
        )

    def query_batch(
        self,
        questions: List[str],
        use_llm: bool = True
    ) -> List[RAGResponse]:
        """
        Query the RAG system with multiple questions

        Args:
            questions: List of questions
            use_llm: Whether to use LLM

        Returns:
            List of RAGResponse objects
        """
        results = []
        for question in questions:
            result = self.query(question, use_llm=use_llm)
            results.append(result)
        return results

    def get_stats(self) -> Dict:
        """
        Get statistics about the knowledge base

        Returns:
            Dict with statistics
        """
        return {
            'total_chunks': len(self.vector_store),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'embedding_model': self.embedding_model,
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold,
            'initialized': self._initialized
        }


class SimpleRAGSystem(RAGSystem):
    """
    Simplified RAG system without threshold filtering for testing.
    """

    def query(self, question: str, use_llm: bool = True) -> str:
        """
        Simple query without threshold filtering

        Args:
            question: User question
            use_llm: Whether to use LLM

        Returns:
            Answer string
        """
        # Get results
        results = self.retriever.retrieve(question, use_threshold=False)

        if not results:
            return "No relevant context found."

        # Format context
        context = self.retriever.format_context(results[:self.top_k])

        if use_llm:
            # Generate response
            llm_result = self.llm.generate_with_sources(
                prompt=question,
                context=context
            )
            return llm_result.text
        else:
            return context


def create_rag_system(
    api_key: str = None,
    **kwargs
) -> RAGSystem:
    """
    Factory function to create RAG system

    Args:
        api_key: Claude API key
        **kwargs: Additional configuration

    Returns:
        Configured RAGSystem instance
    """
    return RAGSystem(api_key=api_key, **kwargs)


if __name__ == "__main__":
    # Test RAG system
    logging.basicConfig(level=logging.INFO)

    # Create system
    rag = RAGSystem()

    # Initialize
    print("Initializing RAG system...")
    rag.initialize()

    # Show stats
    stats = rag.get_stats()
    print(f"\nKnowledge base stats: {stats}")

    # Test query
    print("\nTesting query...")
    response = rag.query("What is machine learning?")

    print(f"\nSuccess: {response.success}")
    print(f"Status: {response.status}")
    print(f"Max similarity: {response.max_similarity:.4f}")
    print(f"\nAnswer:\n{response.answer[:500]}...")
