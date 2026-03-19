"""
RAG System Configuration
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PDFS_DIR = DATA_DIR / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORS_DIR = DATA_DIR / "vectors"

# Chunk settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Retrieval settings
TOP_K = 5

# Threshold settings
SIMILARITY_THRESHOLD = 0.7

# Claude API settings
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
CLAUDE_MAX_TOKENS = 1024
CLAUDE_TEMPERATURE = 0.0

# Nougat settings
NOUGAT_MODEL = "facebook/nougat-base"
NOUGAT_BATCH_SIZE = 1

# Qwen API settings (set your API key here or use environment variable)
QWEN_MODEL = "qwen-plus"
QWEN_API_KEY = "your-qwen-api-key-here"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# LLM provider choice: "qwen" or "claude"
LLM_PROVIDER = "qwen"


class Config:
    """Configuration class for RAG system"""

    # Paths
    PROJECT_ROOT = PROJECT_ROOT
    DATA_DIR = DATA_DIR
    PDFS_DIR = PDFS_DIR
    PROCESSED_DIR = PROCESSED_DIR
    VECTORS_DIR = VECTORS_DIR

    # Chunk settings
    CHUNK_SIZE = CHUNK_SIZE
    CHUNK_OVERLAP = CHUNK_OVERLAP

    # Embedding settings
    EMBEDDING_MODEL = EMBEDDING_MODEL
    EMBEDDING_DIM = EMBEDDING_DIM

    # Retrieval settings
    TOP_K = TOP_K

    # Threshold settings
    SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLD

    # Claude API settings
    CLAUDE_MODEL = CLAUDE_MODEL
    CLAUDE_MAX_TOKENS = CLAUDE_MAX_TOKENS
    CLAUDE_TEMPERATURE = CLAUDE_TEMPERATURE

    # Nougat settings
    NOUGAT_MODEL = NOUGAT_MODEL
    NOUGAT_BATCH_SIZE = NOUGAT_BATCH_SIZE

    # Qwen API settings
    QWEN_MODEL = QWEN_MODEL
    QWEN_API_KEY = QWEN_API_KEY
    QWEN_BASE_URL = QWEN_BASE_URL
    LLM_PROVIDER = LLM_PROVIDER

    @classmethod
    def get_api_key(cls) -> str:
        """Get Claude API key from environment"""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment. "
                "Please set it via: export ANTHROPIC_API_KEY=your_key"
            )
        return api_key
