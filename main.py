"""
Main entry point for RAG system
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from config import Config
from rag_system import RAGSystem, create_rag_system
from benchmark.expertqa_evaluator import ExpertQAEvaluator, run_evaluation


def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demo_query(rag_system: RAGSystem):
    """Run demo queries"""
    print("\n" + "="*60)
    print("RAG System Demo")
    print("="*60)

    # Demo questions (bilingual for Qwen)
    questions = [
        "什么是机器学习?",
        "Explain how neural networks work",
        "What is the time complexity of binary search?",
        "解释快速排序的算法",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-"*40)

        response = rag_system.query(question)

        print(f"Status: {response.status.value}")
        print(f"Success: {response.success}")
        print(f"Max Similarity: {response.max_similarity:.4f}")
        print(f"\nAnswer: {response.answer[:300]}...")

        if response.sources:
            print(f"\nSources:")
            for i, src in enumerate(response.sources[:3]):
                print(f"  [{i+1}] Score: {src['score']:.4f} - {src['source']}")


def interactive_mode(rag_system: RAGSystem):
    """Run in interactive mode"""
    print("\n" + "="*60)
    print("Interactive Mode - Type 'quit' to exit")
    print("="*60)

    while True:
        try:
            question = input("\nQuestion: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                break

            if not question:
                continue

            response = rag_system.query(question)

            print(f"\nAnswer: {response.answer}")

            if response.sources:
                print(f"\nSources (top 3):")
                for i, src in enumerate(response.sources[:3]):
                    print(f"  [{i+1}] {src['source']} (score: {src['score']:.4f})")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


def build_knowledge_base(rag_system: RAGSystem, pdf_dir: Path = None, markdown_dir: Path = None):
    """Build knowledge base"""
    print("\nBuilding knowledge base...")

    pdf_dir = pdf_dir or Config.PDFS_DIR
    markdown_dir = markdown_dir or Config.PROCESSED_DIR

    print(f"PDF directory: {pdf_dir}")
    print(f"Markdown directory: {markdown_dir}")

    rag_system.build_knowledge_base(
        pdf_dir=pdf_dir,
        markdown_dir=markdown_dir
    )

    stats = rag_system.get_stats()
    print(f"\nKnowledge base stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Chunk size: {stats['chunk_size']}")
    print(f"  Embedding model: {stats['embedding_model']}")


def run_benchmark(rag_system: RAGSystem, test_data_path: Path = None):
    """Run ExpertQA benchmark"""
    print("\n" + "="*60)
    print("Running ExpertQA Benchmark")
    print("="*60)

    evaluator = ExpertQAEvaluator(rag_system=rag_system)
    evaluator.load_test_data(test_data_path)
    evaluator.evaluate_all()
    evaluator.print_summary()
    evaluator.save_results()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="RAG System")

    # Mode options
    parser.add_argument(
        '--mode',
        choices=['demo', 'interactive', 'build', 'benchmark'],
        default='demo',
        help='Run mode'
    )

    # Configuration
    parser.add_argument(
        '--api-key',
        help='Claude API key (or set ANTHROPIC_API_KEY env var)'
    )

    parser.add_argument(
        '--embedding-model',
        default=Config.EMBEDDING_MODEL,
        help='Embedding model name'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=Config.CHUNK_SIZE,
        help='Chunk size for text splitting'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=Config.TOP_K,
        help='Number of context chunks to retrieve'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=Config.SIMILARITY_THRESHOLD,
        help='Similarity threshold'
    )

    # Paths
    parser.add_argument(
        '--pdf-dir',
        type=Path,
        help='Directory containing PDF files'
    )

    parser.add_argument(
        '--markdown-dir',
        type=Path,
        help='Directory containing markdown files'
    )

    parser.add_argument(
        '--test-data',
        type=Path,
        help='Path to ExpertQA test data'
    )

    # LLM provider
    parser.add_argument(
        '--llm-provider',
        choices=['qwen', 'claude'],
        default='qwen',
        help='LLM provider (qwen or claude)'
    )

    parser.add_argument(
        '--qwen-model',
        default='qwen-plus',
        help='Qwen model name (qwen-turbo, qwen-plus, qwen-max)'
    )

    # Logging
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(logging.DEBUG if args.debug else logging.INFO)

    # Get API key
    if args.llm_provider == 'qwen':
        api_key = args.api_key or Config.QWEN_API_KEY
    else:
        api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')

    # Create RAG system
    print(f"Initializing RAG system with {args.llm_provider}...")
    rag = create_rag_system(
        api_key=api_key,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
        llm_provider=args.llm_provider
    )

    # Handle different modes
    if args.mode == 'build':
        build_knowledge_base(rag, args.pdf_dir, args.markdown_dir)

    elif args.mode == 'demo':
        # Initialize if needed
        try:
            rag.initialize()
        except Exception as e:
            print(f"Note: {e}")

        # Check if we have a knowledge base
        if rag.vector_store.is_empty():
            print("Knowledge base is empty. Running in build mode first...")
            build_knowledge_base(rag, args.pdf_dir, args.markdown_dir)

        demo_query(rag)

    elif args.mode == 'interactive':
        rag.initialize()
        interactive_mode(rag)

    elif args.mode == 'benchmark':
        rag.initialize()
        run_benchmark(rag, args.test_data)


if __name__ == "__main__":
    main()
