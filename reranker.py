"""
Cross-Encoder Reranker Module.
Re-ranks top vector retrieval results using a Cross-Encoder neural model
to extract exact sentence relevance and filter false positives.
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import CrossEncoder

try:
    from .config import SearchEngineConfig, default_config
except ImportError:
    from config import SearchEngineConfig, default_config

logger = logging.getLogger("CrossEncoderReranker")
logging.basicConfig(level=logging.INFO)


class CrossEncoderReranker:
    """
    Reranks search candidates using a sentence-transformers CrossEncoder.
    Calculates deep cross-attention relevance scores between the query and text candidates.
    """

    def __init__(self, config: Optional[SearchEngineConfig] = None) -> None:
        self.config = config or default_config
        model_name = self.config.reranker_model_name
        logger.info(f"Initializing CrossEncoder with model: '{model_name}'")
        try:
            self.model = CrossEncoder(model_name)
            self._is_mock = False
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder model '{model_name}' ({e}). Using heuristic reranker fallback.")
            self.model = None
            self._is_mock = True

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of retrieved document candidates against the query.
        Returns top_k candidates augmented with 'cross_score' and 'exact_passage'.
        """
        if not candidates:
            return []

        logger.info(f"Reranking {len(candidates)} candidates for query: '{query}'")

        if not self._is_mock and self.model is not None:
            pairs = [[query, cand.get("content", "")] for cand in candidates]
            scores = self.model.predict(pairs)
            for cand, score in zip(candidates, scores):
                cand["cross_score"] = float(score)
        else:
            # Fallback heuristic scoring if model is not loaded
            for cand in candidates:
                cand["cross_score"] = self._heuristic_score(query, cand.get("content", ""))

        # Sort candidates by cross_score descending
        ranked_candidates = sorted(candidates, key=lambda x: x.get("cross_score", 0.0), reverse=True)[:top_k]

        # Extract exact matching sentences/passages from top documents
        for cand in ranked_candidates:
            cand["exact_passage"] = self.extract_exact_passage(query, cand.get("content", ""))

        return ranked_candidates

    def _heuristic_score(self, query: str, content: str) -> float:
        """
        Calculates fallback overlap and exact phrase relevance score.
        """
        q_words = set(re.findall(r'\b\w+\b', query.lower()))
        c_words = set(re.findall(r'\b\w+\b', content.lower()))
        if not q_words:
            return 0.0
        overlap = len(q_words.intersection(c_words)) / len(q_words)

        # Bonus for exact sub-phrase match
        bonus = 0.5 if query.lower() in content.lower() else 0.0
        return float(overlap + bonus)

    def extract_exact_passage(self, query: str, content: str) -> str:
        """
        Extracts the most relevant sentence from the document content for context citation.
        """
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', content) if s.strip()]
        if not sentences:
            return content

        best_sentence = sentences[0]
        best_overlap = -1

        q_words = set(re.findall(r'\b\w+\b', query.lower()))

        for sent in sentences:
            s_words = set(re.findall(r'\b\w+\b', sent.lower()))
            overlap = len(q_words.intersection(s_words))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sent

        return best_sentence


if __name__ == "__main__":
    reranker = CrossEncoderReranker()
    sample_query = "Erro PGFN-500 porta 8443"
    candidates = [
        {"id": 1, "content": "O Erro PGFN-500 ocorre na porta 8443 do certificado.", "score": 0.8},
        {"id": 2, "content": "Acesse a rede corporativa e valide a conexao de IP.", "score": 0.75},
        {"id": 3, "content": "Manual do SIPAF na porta 8443 para relatorios.", "score": 0.70}
    ]
    results = reranker.rerank(sample_query, candidates, top_k=2)
    for r in results:
        print(f"Doc ID: {r['id']} | Cross Score: {r['cross_score']:.4f} | Passage: {r['exact_passage']}")
