"""
Main Execution Script for Motor de Busca Contextual Híbrido (Qdrant).
Demonstrates end-to-end pipeline: Ingestion -> LLM Query Routing -> Qdrant Hybrid Search -> Cross-Encoder Reranking -> RAG Answer Synthesis.
"""

import logging
from typing import List
from qdrant_client import QdrantClient

try:
    from .config import SearchEngineConfig, default_config
    from .ingestion import QdrantIngestionPipeline, SAMPLE_DOCUMENTS
    from .hybrid_search import QdrantHybridSearchEngine
    from .rag_synthesizer import RAGSynthesizer
except ImportError:
    from config import SearchEngineConfig, default_config
    from ingestion import QdrantIngestionPipeline, SAMPLE_DOCUMENTS
    from hybrid_search import QdrantHybridSearchEngine
    from rag_synthesizer import RAGSynthesizer

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MainPipeline")


def run_demo() -> None:
    """
    Runs the complete contextual search pipeline over a set of test queries.
    """
    logger.info("==========================================================================")
    logger.info("     STARTING MOTOR DE BUSCA CONTEXTUAL HÍBRIDO (QDRANT UNIFIED)          ")
    logger.info("==========================================================================")

    # 1. Initialize shared in-memory Qdrant client
    config = SearchEngineConfig(qdrant_url=":memory:", collection_name="contextual_demo_kb")
    client = QdrantClient(config.qdrant_url)

    # 2. Run Ingestion Pipeline
    logger.info("\n--- STEP 1: PIPELINE DE INGESTÃO UNIFICADA NO QDRANT ---")
    ingestion = QdrantIngestionPipeline(client=client, config=config)
    ingestion.ingest_documents(SAMPLE_DOCUMENTS)

    # 3. Initialize Search Engine & RAG Synthesizer
    engine = QdrantHybridSearchEngine(client=client, config=config)
    synthesizer = RAGSynthesizer(config=config)

    # 4. Test Queries showcasing different intent types
    demo_queries: List[str] = [
        "Erro PGFN-500 ao enviar declaracao",
        "Qual a porta do servidor SIPAF?",
        "Como resolver falha de acesso na rede corporativa?",
        "Como funciona a autenticacao 2FA no Keycloak?"
    ]

    for idx, query in enumerate(demo_queries, start=1):
        print("\n" + "=" * 80)
        print(f" DEMO QUERY [{idx}/{len(demo_queries)}]: \"{query}\"")
        print("=" * 80)

        # Run hybrid search
        search_output = engine.search(query)
        analysis = search_output["query_analysis"]

        print(f"\n🧠 [1] LLM QUERY ANALYZER & ROTEADOR DE CONTEXTO:")
        print(f"    - Intenção Detectada : {analysis.intent_type.upper()}")
        print(f"    - Peso Léxico (Sparse): {analysis.sparse_weight:.2f}")
        print(f"    - Peso Semântico (Dense): {analysis.dense_weight:.2f}")
        print(f"    - Palavras/Códigos   : {analysis.detected_keywords}")
        print(f"    - Raciocínio LLM     : {analysis.reasoning}")

        print(f"\n🔍 [2] CANDIDATOS RECUPERADOS NO QDRANT (TOP FUSION):")
        for cand in search_output["candidates"][:3]:
            print(
                f"    - Doc #{cand['id']} | Fusion Score: {cand['fusion_score']:.4f} "
                f"(Dense Norm: {cand['norm_dense']:.2f}, Sparse Norm: {cand['norm_sparse']:.2f}) | "
                f"Título: {cand['payload'].get('title')}"
            )

        print(f"\n⚡ [3] CROSS-ENCODER RERANKING (REORDENAÇÃO NEURAL):")
        for item in search_output["reranked_results"]:
            print(
                f"    - Doc #{item['id']} | Cross Score: {item['cross_score']:.4f} | "
                f"Trecho Exato: \"{item['exact_passage']}\""
            )

        # Run RAG Synthesis
        rag_output = synthesizer.synthesize(query, search_output)

        print(f"\n📝 [4] SÍNTESE DA RESPOSTA FINAL (RAG):")
        print("-" * 80)
        print(rag_output["answer"])
        print("-" * 80)

    logger.info("\n==========================================================================")
    logger.info("     DEMO CONCLUÍDA COM SUCESSO! MOTOR DE BUSCA OPERACIONAL.              ")
    logger.info("==========================================================================")


if __name__ == "__main__":
    run_demo()
