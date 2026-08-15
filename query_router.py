"""
LLM Query Analyzer & Context Router.
Analyzes user queries to determine the search strategy and assign dynamic weights
between Lexical Sparse Search (SPLADE/BM25) and Semantic Dense Search.
"""

import json
import logging
import re
from typing import List, Optional
from pydantic import BaseModel, Field

try:
    from .config import SearchEngineConfig, default_config
except ImportError:
    from config import SearchEngineConfig, default_config

logger = logging.getLogger("LLMQueryAnalyzer")
logging.basicConfig(level=logging.INFO)


class QueryAnalysisResult(BaseModel):
    query: str = Field(description="Original user query")
    intent_type: str = Field(
        description="Type of query intent: 'exact_technical', 'conceptual_semantic', or 'hybrid_balanced'"
    )
    dense_weight: float = Field(
        description="Weight allocated to dense semantic vector search (0.0 to 1.0)"
    )
    sparse_weight: float = Field(
        description="Weight allocated to sparse lexical vector search (0.0 to 1.0)"
    )
    detected_keywords: List[str] = Field(
        default_factory=list, description="Specific terms, acronyms, or codes detected in query"
    )
    reasoning: str = Field(description="Explanation of why these weights were assigned")


SYSTEM_PROMPT = """Você é um Engenheiro de Busca Sênior e Roteador de Contexto para um Motor de Busca Híbrido (Qdrant).
Sua missão é analisar a intenção da pergunta do usuário e definir os pesos para Busca Semântica (dense_weight) e Busca Léxica/Palavras-chave (sparse_weight).

REGRAS RIGOROSAS DE PESO:
1. Se a pergunta contiver CÓDIGOS DE ERRO, SIGLAS TÉCNICAS, NÚMEROS DE PORTA OU IDs (ex: "Erro PGFN-500", "porta 8443", "SIPAF", "SIGA-2024"):
   - Atribua Alta Busca Léxica: sparse_weight = 0.8 e dense_weight = 0.2
   - intent_type = "exact_technical"
2. Se a pergunta for CONCEITUAL, DÚVIDA OU DICA DE COMO FAZER (ex: "como resolver falha de acesso na rede", "como redefinir minha senha"):
   - Atribua Alta Busca Semântica: dense_weight = 0.8 e sparse_weight = 0.2
   - intent_type = "conceptual_semantic"
3. Se a pergunta misturar termos técnicos com explicações amplas:
   - Atribua Pesos Equilibrados: dense_weight = 0.5 e sparse_weight = 0.5
   - intent_type = "hybrid_balanced"

Gere SEMPRE um JSON válido com o formato:
{
  "query": "<query do usuário>",
  "intent_type": "<exact_technical|conceptual_semantic|hybrid_balanced>",
  "dense_weight": 0.2,
  "sparse_weight": 0.8,
  "detected_keywords": ["PGFN-500", "8443"],
  "reasoning": "<explicação resumida>"
}
"""


class LLMQueryAnalyzer:
    """
    LLM Query Router that evaluates user query intent and dynamically assigns
    dense_weight and sparse_weight for Qdrant hybrid retrieval.
    """

    def __init__(self, config: Optional[SearchEngineConfig] = None) -> None:
        self.config = config or default_config

    def analyze(self, query: str) -> QueryAnalysisResult:
        """
        Analyzes the query using OpenAI, Ollama, or heuristic fallback if offline.
        """
        logger.info(f"Analyzing query intent for: '{query}'")

        # 1. Try OpenAI if API key available
        if self.config.openai_api_key:
            try:
                return self._analyze_openai(query)
            except Exception as e:
                logger.warning(f"OpenAI analysis failed ({e}). Falling back to heuristic/mock analyzer.")

        # 2. Try Heuristic Rule Analyzer (fast, deterministic, offline-resilient)
        return self._analyze_heuristic(query)

    def _analyze_openai(self, query: str) -> QueryAnalysisResult:
        import openai

        client = openai.OpenAI(api_key=self.config.openai_api_key)
        response = client.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        result = QueryAnalysisResult(**data)
        logger.info(
            f"LLM Router Result -> Intent: '{result.intent_type}', "
            f"Weights: [Dense={result.dense_weight}, Sparse={result.sparse_weight}]. "
            f"Reasoning: {result.reasoning}"
        )
        return result

    def _analyze_heuristic(self, query: str) -> QueryAnalysisResult:
        """
        Robust heuristic rule analyzer that evaluates codes, acronyms, and technical terms.
        """
        # Specific patterns for error codes (e.g. PGFN-500, SIGA-2024), ports (e.g. 8443, 9090), and technical acronyms
        code_pattern = re.compile(r'\b([A-Z]{2,}-[0-9]{3,}|porta\s+\d+|[0-9]{4})\b', re.IGNORECASE)
        code_matches = code_pattern.findall(query)

        # Strict list of technical acronyms / identifiers
        strict_tech_tokens = {"pgfn", "sipaf", "siga", "jwt", "2fa", "dhcp", "dns", "8443", "9090", "500", "2024"}
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        matched_tokens = sorted(list(query_words.intersection(strict_tech_tokens).union(set(code_matches))))

        # Conceptual indicators
        conceptual_indicators = ["como", "por que", "qual", "resolver", "solucionar", "procedimento", "passo a passo", "explicar"]
        is_conceptual = any(ind in query.lower() for ind in conceptual_indicators)

        if matched_tokens and not is_conceptual:
            intent_type = "exact_technical"
            sparse_weight = 0.8
            dense_weight = 0.2
            reasoning = f"Detectados códigos/siglas rígidas {matched_tokens}. Priorizando busca léxica/esparsa."
        elif matched_tokens and is_conceptual:
            intent_type = "hybrid_balanced"
            sparse_weight = 0.5
            dense_weight = 0.5
            reasoning = f"Detectados termos técnicos ({matched_tokens}) combinados com pergunta explicativa. Pesos equilibrados."
        else:
            intent_type = "conceptual_semantic"
            sparse_weight = 0.2
            dense_weight = 0.8
            reasoning = "Pergunta conceitual/interpretativa sem códigos rígidos. Priorizando busca semântica/densa."

        result = QueryAnalysisResult(
            query=query,
            intent_type=intent_type,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            detected_keywords=list(set(matched_tokens)),
            reasoning=reasoning
        )
        logger.info(
            f"Router (Heuristic) Result -> Intent: '{result.intent_type}', "
            f"Weights: [Dense={result.dense_weight}, Sparse={result.sparse_weight}]. "
            f"Reasoning: {result.reasoning}"
        )
        return result


if __name__ == "__main__":
    router = LLMQueryAnalyzer()

    test_queries = [
        "Erro PGFN-500 ao enviar declaracao",
        "Qual a porta do servidor SIPAF?",
        "Como resolver falha de acesso na rede corporativa?",
        "O que e o sistema Keycloak e como funciona a autenticacao 2FA?"
    ]

    for q in test_queries:
        res = router.analyze(q)
        print("-" * 60)
        print(f"Query: {res.query}")
        print(f"Intent: {res.intent_type} | Weights: Dense={res.dense_weight}, Sparse={res.sparse_weight}")
        print(f"Keywords: {res.detected_keywords}")
        print(f"Reasoning: {res.reasoning}")
