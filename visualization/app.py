"""
Streamlit Web App for RAG System Visualization
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rag_system import RAGSystem
from config import Config


def init_session_state():
    """Initialize session state"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def get_rag_system():
    """Get or create RAG system"""
    if st.session_state.rag_system is None:
        with st.spinner('Loading RAG system...'):
            rag = RAGSystem(llm_provider='qwen', similarity_threshold=0.5)
            rag.initialize()
            st.session_state.rag_system = rag
    return st.session_state.rag_system


def show_knowledge_base_stats(rag):
    """Display knowledge base statistics"""
    st.header("📚 Knowledge Base Statistics")

    stats = rag.get_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Chunks", f"{stats['total_chunks']:,}")

    with col2:
        st.metric("Chunk Size", stats['chunk_size'])

    with col3:
        st.metric("Top-K", stats['top_k'])

    with col4:
        st.metric("Threshold", stats['similarity_threshold'])


def show_retrieval_visualization(query, results):
    """Show retrieval visualization"""
    st.subheader("🔍 Retrieval Process")

    # Query info
    st.markdown(f"**Query:** {query}")

    # Results as table
    if results:
        data = []
        for r in results:
            data.append({
                'Rank': r.rank,
                'Score': f"{r.score:.4f}",
                'Source': r.metadata.get('source', 'Unknown'),
                'Text Preview': r.text[:150] + '...'
            })

        df = pd.DataFrame(data)

        # Display with highlighting
        st.dataframe(
            df,
            column_config={
                'Score': st.column_config.ProgressColumn(
                    'Similarity Score',
                    format='%.4f',
                    min_value=0,
                    max_value=1,
                ),
            },
            use_container_width=True
        )

        # Score distribution chart
        scores = [r.score for r in results]

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.barh(range(len(scores)), scores, color='steelblue')
        ax.set_yticks(range(len(scores)))
        ax.set_yticklabels([f"Source {i+1}" for i in range(len(scores))])
        ax.set_xlabel('Similarity Score')
        ax.set_title('Retrieval Scores')
        ax.set_xlim(0, 1)

        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.02, i, f'{score:.4f}', va='center')

        st.pyplot(fig)


def show_answer_result(response):
    """Show answer result"""
    st.subheader("💬 Answer")

    # Status indicator
    if response.success:
        st.success(f"✅ Answer generated (Status: {response.status.value})")
    else:
        st.error(f"❌ {response.message}")

    # Answer
    st.markdown(response.answer)

    # Metadata
    with st.expander("📊 Response Metadata"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Max Similarity", f"{response.max_similarity:.4f}")
        with col2:
            st.metric("Sources Found", len(response.sources))


def show_source_details(response):
    """Show source details"""
    if response.sources:
        st.subheader("📄 Source Details")

        for i, src in enumerate(response.sources):
            with st.expander(f"Source {i+1}: {src['source']} (Score: {src['score']:.4f})"):
                st.text(src['text'])


def main():
    """Main app"""
    st.set_page_config(
        page_title="RAG Knowledge System",
        page_icon="📚",
        layout="wide"
    )

    init_session_state()

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # Settings
    threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )

    top_k = st.sidebar.slider(
        "Top-K Results",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

    # Main content
    st.title("📚 RAG Knowledge System")
    st.markdown("""
    This is a **Retrieval-Augmented Generation** system for computer science knowledge.
    Ask questions and get accurate answers based on the knowledge base.
    """)

    # Initialize RAG system with settings
    if st.session_state.rag_system is None:
        rag = RAGSystem(
            llm_provider='qwen',
            similarity_threshold=threshold,
            top_k=top_k
        )
        rag.initialize()
        st.session_state.rag_system = rag
    else:
        # Update settings
        st.session_state.rag_system.similarity_threshold = threshold
        st.session_state.rag_system.top_k = top_k
        st.session_state.rag_system.retriever.similarity_threshold = threshold
        st.session_state.rag_system.retriever.top_k = top_k

    # Show stats
    show_knowledge_base_stats(st.session_state.rag_system)

    st.divider()

    # Chat interface
    st.subheader("💬 Ask a Question")

    # Input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is machine learning?",
        key="query_input"
    )

    # Search button
    if st.button("🔎 Get Answer", type="primary"):
        if query:
            with st.spinner('Processing...'):
                # Get response
                response = st.session_state.rag_system.query(query)

                # Add to history
                st.session_state.chat_history.append({
                    'query': query,
                    'response': response
                })

    # Show results
    if st.session_state.chat_history:
        st.divider()
        st.subheader("📝 Results")

        # Show latest result
        latest = st.session_state.chat_history[-1]

        # Show retrieval visualization
        show_retrieval_visualization(
            latest['query'],
            latest['response'].evaluation.results if latest['response'].evaluation else []
        )

        # Show answer
        show_answer_result(latest['response'])

        # Show sources
        show_source_details(latest['response'])

    # Chat history
    if len(st.session_state.chat_history) > 1:
        st.divider()
        with st.expander("📜 Chat History"):
            for i, chat in enumerate(st.session_state.chat_history[:-1]):
                st.markdown(f"**Q{i+1}:** {chat['query']}")
                st.markdown(f"**A{i+1}:** {chat['response'].answer[:200]}...")
                st.divider()


if __name__ == "__main__":
    main()
