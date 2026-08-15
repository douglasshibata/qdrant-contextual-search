# 🔎 Motor de Busca Contextual Híbrido (Qdrant + FastEmbed + LLM Router + Cross-Encoder)

> **Plataforma Enterprise de Busca Híbrida Unificada (Dense + Sparse Vectors) e RAG de Última Geração**

---

## 1. 🎯 Cabeçalho & Visão Geral

O **Motor de Busca Contextual Híbrido** é uma solução avançada de recuperação de informação e RAG (Retrieval-Augmented Generation) projetada para eliminar a necessidade de manter dois bancos de dados separados (como OpenSearch/Elasticsearch para palavras-chave e Qdrant/Pinecone para vetores semânticos).

A grande inovação desta arquitetura baseia-se em dois pilares:
1. **Unificação Vetorial no Qdrant:** Utilização da funcionalidade nativa de coleções com múltiplos vetores (`text-dense` para semântica via FastEmbed BGE e `text-sparse` para busca léxica via SPLADE/BM25).
2. **LLM como Roteador de Contexto (Dynamic Weighting):** Antes da execução da busca, um módulo LLM analisa a intenção do usuário. Se a pergunta contiver códigos de erro rígidos ou siglas (ex: `"Erro PGFN-500"` ou `"porta 8443"`), o roteador atribui maior peso à busca léxica (`sparse_weight = 0.8`). Se for uma dúvida conceitual/interpretativa (ex: `"como resolver falha de acesso na rede"`), atribui maior peso à busca semântica (`dense_weight = 0.8`).

---

## 2. 🏗️ Arquitetura do Motor Contextual

```
 +------------------------+
 |   Query do Usuário     |  ("Erro PGFN-500 ao enviar declaração")
 +-----------+------------+
             |
             v
 +-----------------------------------------------------------------------+
 |                  LLM Query Analyzer (Context Router)                   |
 |  - Detecta códigos de erro, siglas, portas e intenções conceitual   |
 |  - Atribui Pesos Dinâmicos: [dense_weight: 0.2, sparse_weight: 0.8]  |
 +-----------+-----------------------------------------------------------+
             |
             v
 +-----------------------------------------------------------------------+
 |                       FastEmbed Dual Generator                        |
 |  - Generates Dense Embedding (BAAI/bge-small-en-v1.5, dim 384)        |
 |  - Generates Sparse Embedding (prithivida/Splade_PP_en_v1 / BM25)     |
 +-----------+-----------------------------------------------------------+
             |
             v
 +-----------------------------------------------------------------------+
 |                Qdrant Unified Hybrid Search API                       |
 |  - Executa busca multi-vetorial simultânea na coleção unificada:     |
 |      * 'text-dense' (VectorParams - Cosine)                           |
 |      * 'text-sparse' (SparseVectorParams - SPLADE)                    |
 |  - Aplica Weighted Score Fusion:                                      |
 |    FusionScore = (dense_weight * NormDense) + (sparse_weight * NormSparse) |
 +-----------+-----------------------------------------------------------+
             |  (Top 10 Candidatos)
             v
 +-----------------------------------------------------------------------+
 |                     Cross-Encoder Neural Reranker                     |
 |  - Reordena candidatos com modelo 'cross-encoder/ms-marco-MiniLM-L6' |
 |  - Extrai a frase exata mais relevante de cada documento              |
 +-----------+-----------------------------------------------------------+
             |  (Top 3 Candidatos Validados)
             v
 +-----------------------------------------------------------------------+
 |                     LLM RAG Answer Synthesizer                        |
 |  - Gera resposta clara e precisa citando obrigatoriamente a frase     |
 |    exata e o documento de origem                                      |
 +-----------------------------------------------------------------------+
```

### Componentes Chave da Arquitetura

| Componente | Módulo Python | Descrição / Responsabilidade |
| :--- | :--- | :--- |
| **Configuração** | `config.py` | Configuração tipada (`SearchEngineConfig`) para conexões Qdrant, modelos e chaves |
| **Embedding Duplo** | `embeddings.py` | `HybridEmbedder` usando FastEmbed para vetores Densos e Esparsos |
| **Ingestão Unificada** | `ingestion.py` | `QdrantIngestionPipeline` que cria coleções multi-vetoriais e realiza upsert |
| **Roteador de Contexto** | `query_router.py` | `LLMQueryAnalyzer` com prompt e fallback determinístico para definição de pesos |
| **Reranker Neural** | `reranker.py` | `CrossEncoderReranker` com Cross-Encoder e extrator de frases exatas |
| **Busca Híbrida** | `hybrid_search.py` | `QdrantHybridSearchEngine` orquestrando busca e fusão de scores |
| **Síntese RAG** | `rag_synthesizer.py` | `RAGSynthesizer` produzindo respostas fundamentadas com citação obrigatória |
| **CLI / Runner** | `main.py` | Ponto de entrada interativo executando demonstração completa |

---

## 3. ⚙️ Instalação & Pré-requisitos

### Pré-requisitos

- **Python:** 3.11 ou 3.12+
- **Qdrant:** Servidor local, em nuvem (Qdrant Cloud) ou instância em memória (`:memory:`)

### Instalação de Dependências

```bash
pip install qdrant-client fastembed sentence-transformers openai pydantic pytest
```

---

## 4. 🛠️ Configuração

A configuração pode ser ajustada via variáveis de ambiente ou alterando os campos da classe `SearchEngineConfig` em `config.py`:

| Variável | Valor Padrão | Descrição |
| :--- | :--- | :--- |
| `QDRANT_URL` | `:memory:` | URL do Qdrant (ex: `http://localhost:6333` ou `:memory:`) |
| `QDRANT_API_KEY` | `None` | Chave de API do Qdrant Cloud (se aplicável) |
| `QDRANT_COLLECTION` | `contextual_hybrid_kb` | Nome da coleção no Qdrant |
| `DENSE_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Modelo FastEmbed para vetores densos semânticos |
| `SPARSE_MODEL_NAME` | `prithivida/Splade_PP_en_v1` | Modelo FastEmbed para vetores esparsos léxicos (SPLADE/BM25) |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Modelo Cross-Encoder para reranking neural |
| `OPENAI_API_KEY` | `None` | Chave de API para rotas e síntese GPT-4o (opcional) |
| `OPENAI_MODEL` | `gpt-4o` | Modelo OpenAI para análise e síntese |

---

## 5. 🚀 Execução & Uso

### Executando a Demonstração CLI Completa

```bash
python3 -m qdrant-contextual-search.main
```

### Exemplo de Uso em Código Python

```python
from qdrant_client import QdrantClient
from qdrant-contextual-search.config import SearchEngineConfig
from qdrant-contextual-search.ingestion import QdrantIngestionPipeline, SAMPLE_DOCUMENTS
from qdrant-contextual-search.hybrid_search import QdrantHybridSearchEngine
from qdrant-contextual-search.rag_synthesizer import RAGSynthesizer

# 1. Conectar ao Qdrant em memória
client = QdrantClient(":memory:")

# 2. Ingerir documentos com vetores Densos e Esparsos
pipeline = QdrantIngestionPipeline(client=client)
pipeline.ingest_documents(SAMPLE_DOCUMENTS)

# 3. Executar busca híbrida contextual
engine = QdrantHybridSearchEngine(client=client)
search_results = engine.search("Erro PGFN-500 ao enviar declaracao")

# 4. Sintetizar resposta RAG com citação de frase exata
synthesizer = RAGSynthesizer()
rag_response = synthesizer.synthesize("Erro PGFN-500 ao enviar declaracao", search_results)

print(rag_response["answer"])
```

---

## 6. 🧪 Testes Automatizados

O repositório inclui uma suíte completa de testes unitários e de integração utilizando `pytest`.

Para executar todos os testes:

```bash
python3 -m pytest qdrant-contextual-search/tests/
```

---

## 7. 🔒 Segurança & Melhores Práticas

1. **Gestão de Segredos:** Nunca exponha a `OPENAI_API_KEY` ou `QDRANT_API_KEY` diretamente no código. Utilize variáveis de ambiente ou um cofre de segredos.
2. **Isolamento de Coleções:** Utilize nomes de coleções dedicados por tenant ou ambiente (desenvolvimento vs produção).
3. **Resiliência e Fallbacks:** O sistema inclui fallbacks determinísticos para que a busca híbrida continue operando com máxima precisão mesmo quando serviços externos de LLM estiverem indisponíveis.

---

## 8. 🚨 Troubleshooting & FAQ

### 1. `AttributeError: 'QdrantClient' object has no attribute 'search'`
**Causa:** Na versão 1.19+ do `qdrant-client`, os métodos de busca utilizam a API unificada `query_points`.
**Solução:** O código já está atualizado para utilizar `client.query_points(collection_name=..., query=..., using=...)`.

### 2. Downloads de Modelos FastEmbed / Sentence-Transformers demorados na primeira execução
**Causa:** Na primeira inicialização, os modelos HuggingFace são baixados e armazenados em cache local (`~/.cache/huggingface`). As execuções subsequentes utilizarão os arquivos em cache.
