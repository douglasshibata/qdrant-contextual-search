"""
RAG Answer Synthesizer.
Synthesizes precise answers from Top Cross-Encoder reranked results,
mandating exact phrase citations and source metadata attribution.
"""

import logging
from typing import Dict, Any, List, Optional

try:
    from .config import SearchEngineConfig, default_config
except ImportError:
    from config import SearchEngineConfig, default_config

logger = logging.getLogger("RAGSynthesizer")
logging.basicConfig(level=logging.INFO)


SYNTHESIS_SYSTEM_PROMPT = """Você é um Assistente de Conhecimento Técnico RAG (Retrieval-Augmented Generation) de alta precisão.
Sua tarefa é responder à Pergunta do Usuário utilizando EXCLUSIVAMENTE os contextos fornecidos recuperados do Motor de Busca Híbrido.

REGRAS RIGOROSAS:
1. Responda diretamente e fundamentando cada afirmação no contexto recuperado.
2. CITE OBRIGATORIAMENTE a frase exata ou documento de onde extraiu a solução (ex: "Conforme o 'Manual de Erros Fiscais - Código PGFN-500': 'O Erro PGFN-500 ocorre quando...'").
3. Se o contexto não contiver a resposta para a pergunta, declare claramente que não há informações suficientes no banco de dados.
4. Mantenha tom profissional, objetivo e estruturado.
"""


class RAGSynthesizer:
    """
    Synthesizes grounded RAG responses using top Cross-Encoder validated search results.
    """

    def __init__(self, config: Optional[SearchEngineConfig] = None) -> None:
        self.config = config or default_config

    def synthesize(self, query: str, search_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates grounded response using Top 3 reranked contexts.
        """
        logger.info(f"Synthesizing RAG response for query: '{query}'")

        reranked = search_result.get("reranked_results", [])
        if not reranked:
            return {
                "answer": "Nenhum documento relevante foi encontrado no banco de dados para responder a esta pergunta.",
                "cited_sources": [],
                "grounded": False
            }

        # Build context prompt
        context_blocks: List[str] = []
        cited_sources: List[Dict[str, Any]] = []

        for idx, item in enumerate(reranked, start=1):
            title = item.get("payload", {}).get("title", f"Documento {idx}")
            system = item.get("payload", {}).get("system", "Geral")
            passage = item.get("exact_passage", "")
            content = item.get("content", "")

            block = (
                f"--- CONTEXTO [{idx}] ---\n"
                f"Título: {title}\n"
                f"Sistema/Módulo: {system}\n"
                f"Trecho Principal: \"{passage}\"\n"
                f"Conteúdo Completo: {content}\n"
            )
            context_blocks.append(block)
            cited_sources.append({
                "doc_id": item["id"],
                "title": title,
                "system": system,
                "exact_passage": passage,
                "cross_score": item.get("cross_score", 0.0)
            })

        formatted_context = "\n".join(context_blocks)

        # Generate response using OpenAI if key exists, else deterministic fallback
        if self.config.openai_api_key:
            try:
                answer = self._generate_openai(query, formatted_context)
            except Exception as e:
                logger.warning(f"OpenAI synthesis failed ({e}). Using grounded template synthesizer.")
                answer = self._generate_template(query, cited_sources)
        else:
            answer = self._generate_template(query, cited_sources)

        return {
            "answer": answer,
            "cited_sources": cited_sources,
            "query_analysis": search_result.get("query_analysis"),
            "grounded": True
        }

    def _generate_openai(self, query: str, context: str) -> str:
        import openai

        client = openai.OpenAI(api_key=self.config.openai_api_key)
        user_prompt = f"Pergunta do Usuário: {query}\n\nContextos Recuperados:\n{context}\n\nResposta Fundamentada:"
        response = client.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content or "Sem resposta gerada pelo modelo."

    def _generate_template(self, query: str, cited_sources: List[Dict[str, Any]]) -> str:
        """
        Grounded response synthesizer when offline or without API key.
        """
        top_source = cited_sources[0]
        lines = [
            f"Com base nas informações recuperadas do banco de conhecimento contextual:\n",
            f"**Resposta Directa:**",
            f"Para a consulta \"{query}\", a orientação identificada é:\n",
            f"> \"{top_source['exact_passage']}\"\n",
            f"**Fonte e Contexto Recuperado:**",
            f"- **Documento:** {top_source['title']} (Módulo {top_source['system']})",
            f"- **Relevância Neural (Cross-Encoder):** {top_source['cross_score']:.4f}\n"
        ]

        if len(cited_sources) > 1:
            lines.append("**Fontes Adicionais Validadas:**")
            for src in cited_sources[1:]:
                lines.append(f"- *{src['title']}*: \"{src['exact_passage']}\"")

        return "\n".join(lines)


if __name__ == "__main__":
    from .hybrid_search import QdrantHybridSearchEngine
    from .ingestion import QdrantIngestionPipeline, SAMPLE_DOCUMENTS
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    pipeline = QdrantIngestionPipeline(client=client)
    pipeline.ingest_documents(SAMPLE_DOCUMENTS)

    engine = QdrantHybridSearchEngine(client=client)
    search_res = engine.search("Erro PGFN-500 porta 8443")

    synthesizer = RAGSynthesizer()
    rag_out = synthesizer.synthesize("Erro PGFN-500 porta 8443", search_res)

    print("=== RAG SYNTHESIS OUTPUT ===")
    print(rag_out["answer"])
