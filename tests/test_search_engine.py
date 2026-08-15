"""
Unit and Integration Tests for Motor de Busca Contextual Híbrido (Qdrant).
Run with: python3 -m pytest qdrant-contextual-search/tests/
"""

import os
import sys
import pytest
from qdrant_client import QdrantClient

# Add module parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SearchEngineConfig
from embeddings import HybridEmbedder
from ingestion import QdrantIngestionPipeline, SAMPLE_DOCUMENTS
from query_router import LLMQueryAnalyzer, QueryAnalysisResult
from reranker import CrossEncoderReranker
from hybrid_search import QdrantHybridSearchEngine
from rag_synthesizer import RAGSynthesizer


@pytest.fixture
def memory_config() -> SearchEngineConfig:
    return SearchEngineConfig(qdrant_url=":memory:", collection_name="test_contextual_kb")


@pytest.fixture
def memory_client(memory_config: SearchEngineConfig) -> QdrantClient:
    return QdrantClient(memory_config.qdrant_url)


def test_hybrid_embedder():
    """Test FastEmbed dense and sparse embedding generation."""
    embedder = HybridEmbedder()
    dense, sparse = embedder.embed_single("Erro PGFN-500 porta 8443")
    assert len(dense) == 384
    assert len(sparse.indices) > 0
    assert len(sparse.values) == len(sparse.indices)


def test_qdrant_ingestion_pipeline(memory_client: QdrantClient, memory_config: SearchEngineConfig):
    """Test multi-vector Qdrant collection setup and document ingestion."""
    pipeline = QdrantIngestionPipeline(client=memory_client, config=memory_config)
    points_count = pipeline.ingest_documents(SAMPLE_DOCUMENTS)
    assert points_count == len(SAMPLE_DOCUMENTS)

    col_info = memory_client.get_collection(memory_config.collection_name)
    assert col_info.points_count == len(SAMPLE_DOCUMENTS)


def test_llm_query_analyzer_exact_technical():
    """Test that queries with technical codes assign high sparse_weight."""
    router = LLMQueryAnalyzer()
    res: QueryAnalysisResult = router.analyze("Erro PGFN-500 ao enviar declaracao")
    assert res.intent_type == "exact_technical"
    assert res.sparse_weight >= 0.7
    assert res.dense_weight <= 0.3
    assert "PGFN-500" in res.detected_keywords or "500" in res.detected_keywords or "pgfn" in res.detected_keywords


def test_llm_query_analyzer_conceptual():
    """Test that conceptual queries assign high dense_weight."""
    router = LLMQueryAnalyzer()
    res: QueryAnalysisResult = router.analyze("Como resolver falha de acesso na rede corporativa?")
    assert res.intent_type == "conceptual_semantic"
    assert res.dense_weight >= 0.7
    assert res.sparse_weight <= 0.3


def test_cross_encoder_reranker():
    """Test Cross-Encoder reranking and exact passage extraction."""
    reranker = CrossEncoderReranker()
    query = "Erro PGFN-500 porta 8443"
    candidates = [
        {"id": 1, "content": "O Erro PGFN-500 ocorre na porta 8443 do certificado.", "score": 0.8},
        {"id": 2, "content": "Acesse a rede corporativa e valide a conexao de IP.", "score": 0.75}
    ]
    reranked = reranker.rerank(query, candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["id"] == 1
    assert "exact_passage" in reranked[0]
    assert "Erro PGFN-500" in reranked[0]["exact_passage"]


def test_hybrid_search_engine_integration(memory_client: QdrantClient, memory_config: SearchEngineConfig):
    """Test full search execution with score fusion and reranking."""
    # Setup database
    pipeline = QdrantIngestionPipeline(client=memory_client, config=memory_config)
    pipeline.ingest_documents(SAMPLE_DOCUMENTS)

    engine = QdrantHybridSearchEngine(client=memory_client, config=memory_config)

    # Technical code query
    res_tech = engine.search("Erro PGFN-500")
    assert len(res_tech["reranked_results"]) > 0
    top_tech_id = res_tech["reranked_results"][0]["id"]
    assert top_tech_id == 1  # Manual PGFN-500

    # Conceptual query
    res_concept = engine.search("Como resolver falha de acesso na rede corporativa?")
    assert len(res_concept["reranked_results"]) > 0
    top_concept_id = res_concept["reranked_results"][0]["id"]
    assert top_concept_id == 3  # Conectividade de Rede


def test_rag_synthesizer_grounding(memory_client: QdrantClient, memory_config: SearchEngineConfig):
    """Test RAG synthesis output formatting and citations."""
    pipeline = QdrantIngestionPipeline(client=memory_client, config=memory_config)
    pipeline.ingest_documents(SAMPLE_DOCUMENTS)

    engine = QdrantHybridSearchEngine(client=memory_client, config=memory_config)
    search_res = engine.search("Qual a porta do servidor SIPAF?")

    synthesizer = RAGSynthesizer(config=memory_config)
    rag_out = synthesizer.synthesize("Qual a porta do servidor SIPAF?", search_res)

    assert rag_out["grounded"] is True
    assert len(rag_out["cited_sources"]) > 0
    assert "8443" in rag_out["answer"] or "SIPAF" in rag_out["answer"]
