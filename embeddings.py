"""
Dual Embedding Generator using FastEmbed.
Generates Dense Vectors (Semantic Context) and Sparse Vectors (Lexical SPLADE/BM25) simultaneously.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from qdrant_client import models
from fastembed import TextEmbedding, SparseTextEmbedding

try:
    from .config import SearchEngineConfig, default_config
except ImportError:
    from config import SearchEngineConfig, default_config

logger = logging.getLogger("HybridEmbedder")
logging.basicConfig(level=logging.INFO)


class HybridEmbedder:
    """
    Generates both dense and sparse embeddings using FastEmbed models.
    Dense model handles semantic meaning while Sparse model (SPLADE/BM25) handles exact keyword matches.
    """

    def __init__(self, config: Optional[SearchEngineConfig] = None) -> None:
        self.config = config or default_config
        logger.info(
            f"Initializing HybridEmbedder with Dense model: '{self.config.dense_model_name}' "
            f"and Sparse model: '{self.config.sparse_model_name}'"
        )
        self._dense_model = TextEmbedding(model_name=self.config.dense_model_name)
        self._sparse_model = SparseTextEmbedding(model_name=self.config.sparse_model_name)

    def embed_single(self, text: str) -> Tuple[List[float], models.SparseVector]:
        """
        Generates dense and sparse embeddings for a single text input.
        """
        dense_vecs, sparse_vecs = self.embed_batch([text])
        return dense_vecs[0], sparse_vecs[0]

    def embed_batch(self, texts: List[str]) -> Tuple[List[List[float]], List[models.SparseVector]]:
        """
        Generates dense and sparse embeddings for a list of texts in batch.
        """
        if not texts:
            return [], []

        logger.debug(f"Generating dense & sparse embeddings for batch of size {len(texts)}")

        # Dense embedding generation
        dense_generator = self._dense_model.embed(texts)
        dense_embeddings = [vector.tolist() if hasattr(vector, "tolist") else list(vector) for vector in dense_generator]

        # Sparse embedding generation
        sparse_generator = self._sparse_model.embed(texts)
        sparse_embeddings: List[models.SparseVector] = []

        for sparse_output in sparse_generator:
            # FastEmbed SparseEmbedding returns objects with .indices and .values
            indices = list(sparse_output.indices)
            values = list(sparse_output.values)
            sparse_embeddings.append(
                models.SparseVector(
                    indices=indices,
                    values=[float(v) for v in values]
                )
            )

        return dense_embeddings, sparse_embeddings


if __name__ == "__main__":
    embedder = HybridEmbedder()
    sample_text = "Erro PGFN-500 ao acessar a porta 8443 do servidor SIPAF"
    dense, sparse = embedder.embed_single(sample_text)
    print(f"Sample text: '{sample_text}'")
    print(f"Dense vector size: {len(dense)}")
    print(f"Sparse vector indices count: {len(sparse.indices)}, values count: {len(sparse.values)}")
