"""
Configuration module for Qdrant Contextual Search Engine.
Handles database connections, embedding models, LLM router specs, and Cross-Encoder settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchEngineConfig:
    # Qdrant Database Settings
    qdrant_url: str = field(
        default_factory=lambda: os.getenv("QDRANT_URL", ":memory:")
    )
    qdrant_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY", None)
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "contextual_hybrid_kb")
    )
    dense_vector_name: str = "text-dense"
    sparse_vector_name: str = "text-sparse"
    dense_vector_size: int = 384  # Dimension for BAAI/bge-small-en-v1.5

    # FastEmbed Models
    dense_model_name: str = field(
        default_factory=lambda: os.getenv("DENSE_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    )
    sparse_model_name: str = field(
        default_factory=lambda: os.getenv("SPARSE_MODEL_NAME", "prithivida/Splade_PP_en_v1")
    )

    # Cross-Encoder Reranking
    reranker_model_name: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )

    # LLM Router & Synthesis Settings
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", None)
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    use_mock_llm_if_no_key: bool = True

    # Search Limits
    top_k_retrieval: int = 10
    top_k_rerank: int = 3


# Global default configuration instance
default_config = SearchEngineConfig()
