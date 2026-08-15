"""
Unified Ingestion Pipeline for Qdrant Hybrid Collection.
Configures multi-vector collection (Dense + Sparse) and upserts document payloads.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models

try:
    from .config import SearchEngineConfig, default_config
    from .embeddings import HybridEmbedder
except ImportError:
    from config import SearchEngineConfig, default_config
    from embeddings import HybridEmbedder

logger = logging.getLogger("QdrantIngestionPipeline")
logging.basicConfig(level=logging.INFO)


SAMPLE_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "title": "Manual de Erros Fiscais - Código PGFN-500",
        "content": "O Erro PGFN-500 ocorre quando há falha de comunicação entre o webservice da Procuradoria-Geral da Fazenda Nacional e o módulo de transmissão de dívida ativa. Para resolver, reinicie o serviço pgfn-connector e valide o certificado digital A1 na porta 8443.",
        "category": "tecnico",
        "system": "PGFN",
        "tags": ["PGFN-500", "erro", "certificado", "porta 8443"]
    },
    {
        "id": 2,
        "title": "Arquitetura e Portas do Servidor SIPAF",
        "content": "O servidor SIPAF (Sistema Integrado de Passagens e Alimentação Força) opera prioritariamente na porta 8443 para HTTPS seguro e na porta 9090 para relatórios internos. Certifique-se de que as regras de firewall do ambiente corporativo liberem o tráfego TCP nestas portas.",
        "category": "infraestrutura",
        "system": "SIPAF",
        "tags": ["SIPAF", "porta 8443", "porta 9090", "firewall"]
    },
    {
        "id": 3,
        "title": "Solução de Problemas de Conectividade de Rede",
        "content": "Para resolver falha de acesso na rede corporativa, verifique primeiramente se o endereço IP do host foi atribuído corretamente via DHCP. Em seguida, execute testes de ping para o gateway padrão e verifique as rotas de DNS primário e secundário.",
        "category": "suporte",
        "system": "REDE",
        "tags": ["rede", "conectividade", "ip", "dhcp", "dns"]
    },
    {
        "id": 4,
        "title": "Guia de Autenticação Única e Gestão de Acessos",
        "content": "Problemas de autenticação ou negação de acesso em sistemas internos geralmente decorrem de tokens JWT expirados ou permissões revogadas no perfil do Keycloak. O procedimento de redefinição exige validação em dois fatores (2FA).",
        "category": "seguranca",
        "system": "KEYCLOAK",
        "tags": ["acesso", "autenticacao", "jwt", "2fa"]
    },
    {
        "id": 5,
        "title": "Manual do Sistema SIGA - Código SIGA-2024",
        "content": "A mensagem de alerta SIGA-2024 indica divergência nos parâmetros de homologação do orçamento anual. O responsável financeiro deve aprovar o lote pendente no painel do ordenador de despesas.",
        "category": "financeiro",
        "system": "SIGA",
        "tags": ["SIGA-2024", "orcamento", "homologacao"]
    }
]


class QdrantIngestionPipeline:
    """
    Manages collection setup and document ingestion into Qdrant using unified dense and sparse vectors.
    """

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        embedder: Optional[HybridEmbedder] = None,
        config: Optional[SearchEngineConfig] = None,
    ) -> None:
        self.config = config or default_config
        self.embedder = embedder or HybridEmbedder(config=self.config)

        if client is not None:
            self.client = client
        else:
            logger.info(f"Connecting to Qdrant at '{self.config.qdrant_url}'")
            if self.config.qdrant_url == ":memory:":
                self.client = QdrantClient(":memory:")
            else:
                self.client = QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=self.config.qdrant_api_key
                )

    def setup_collection(self, recreate: bool = True) -> None:
        """
        Creates or updates the Qdrant collection configured for multi-vector hybrid search.
        """
        col_name = self.config.collection_name
        exists = self.client.collection_exists(collection_name=col_name)

        if exists and recreate:
            logger.info(f"Recreating collection '{col_name}'...")
            self.client.delete_collection(collection_name=col_name)
            exists = False

        if not exists:
            logger.info(f"Creating multi-vector collection '{col_name}' with Dense & Sparse specs.")
            self.client.create_collection(
                collection_name=col_name,
                vectors_config={
                    self.config.dense_vector_name: models.VectorParams(
                        size=self.config.dense_vector_size,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    self.config.sparse_vector_name: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                }
            )
            logger.info(f"Collection '{col_name}' created successfully.")

    def ingest_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Processes and upserts documents into Qdrant collection with dense and sparse vectors.
        """
        if not documents:
            logger.warning("No documents provided for ingestion.")
            return 0

        self.setup_collection(recreate=True)
        col_name = self.config.collection_name

        texts = [doc["content"] for doc in documents]
        logger.info(f"Embedding {len(documents)} documents (Dense + Sparse)...")
        dense_vecs, sparse_vecs = self.embedder.embed_batch(texts)

        points: List[models.PointStruct] = []
        for doc, dense_vec, sparse_vec in zip(documents, dense_vecs, sparse_vecs):
            point_id = doc["id"]
            payload = {k: v for k, v in doc.items() if k != "id"}

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        self.config.dense_vector_name: dense_vec,
                        self.config.sparse_vector_name: sparse_vec
                    },
                    payload=payload
                )
            )

        logger.info(f"Upserting {len(points)} points into Qdrant collection '{col_name}'...")
        self.client.upsert(collection_name=col_name, points=points)

        info = self.client.get_collection(collection_name=col_name)
        logger.info(f"Ingestion complete. Total points in '{col_name}': {info.points_count}")
        return info.points_count


if __name__ == "__main__":
    pipeline = QdrantIngestionPipeline()
    count = pipeline.ingest_documents(SAMPLE_DOCUMENTS)
    print(f"Successfully ingested {count} documents into Qdrant.")
