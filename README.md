# 🔬 Hybrid Retrieval Framework

**Autor**: Thiago Pedroso de Jesus  
**Instituição**: Residência em IA - Bacharelado em Inteligência Artificial - UFG

---

## 📖 Sobre o Projeto

Este repositório implementa um **framework modular e extensível para experimentação com sistemas de Retrieval-Augmented Generation (RAG)**, com foco especial em **busca híbrida multi-modal**. O framework permite combinar diferentes estratégias de recuperação de informação (semântica, lexical, baseada em grafos de entidades) de forma flexível e configurável.

### 🎯 Objetivo

Fornecer uma estrutura robusta e flexível para:

- ✅ **Experimentação rápida** com diferentes técnicas de retrieval
- ✅ **Busca híbrida multi-modal** (semantic + lexical + graph)
- ✅ **Avaliação rigorosa** usando métricas padrão de IR (nDCG, MRR, MAP, Recall, Precision)
- ✅ **Comparação sistemática** entre diferentes abordagens
- ✅ **Reprodutibilidade** via configurações YAML declarativas
- ✅ **Extensibilidade** através de arquitetura baseada em interfaces (ABCs)

### 🌟 Características Principais

- **Modular**: Componentes intercambiáveis (vectorizers, indexes, fusers, rerankers)
- **Multi-modal**: Suporte nativo para busca tri-modal (semantic + TF-IDF + entities)
- **Configurável**: Experimentos definidos via YAML (sem código)
- **Extensível**: Fácil adicionar novos retrievers, vectorizers ou estratégias de fusão
- **Eficiente**: Usa FAISS para busca vetorial rápida, com suporte a GPU
- **Reproduzível**: Cache de índices, entidades e resultados de LLM judges
- **Benchmarkável**: Suporte nativo para datasets BEIR (SciFact, FIQA, NFCorpus, SQuAD)

---

## 🏗️ Arquitetura

O framework segue uma **arquitetura modular baseada em interfaces (ABCs)**, onde cada componente tem uma responsabilidade única:

``` mermaid
flowchart TD

    A["EXPERIMENT RUNNER<br>(Orquestra pipeline completo)"]

    A --> B["RETRIEVERS<br>(Coordenadores)"]
    A --> C["DATASETS<br>(BEIR)"]

    B --> D["VECTORIZER<br>(Texto→Vec)"]
    B --> E["INDEX<br>(Busca)"]
    B --> F["FUSION<br>(Combina)"]

    D --> G["ENCODERS<br>(Building)"]
    F --> H["RERANKER<br>(Refina)"]
```

### 📦 Componentes

1. **Vectorizers** (`src/vectorizers/`): Convertem texto em vetores multi-modais
   - Dense (semantic embeddings)
   - TF-IDF (lexical)
   - Tri-Modal (semantic + TF-IDF + entities)
   - Bi-Modal (combinações de 2 modalidades)
   - Graph (apenas entidades)

2. **Indexes** (`src/indexes/`): Estruturas de busca vetorial eficiente
   - FAISS (busca exata e aproximada)
   - Suporte a GPU
   - Persistência (cache)

3. **Retrievers** (`src/retrievers/`): Orquestram pipeline completo
   - Dense (apenas semantic)
   - BM25 (baseline lexical)
   - Hybrid (tri-modal com reranking)
   - DAT (Dynamic Alpha Tuning com LLM judge)
   - Baseline Hybrid (alpha fixo)

4. **Fusion** (`src/fusion/`): Combinam resultados de múltiplas modalidades
   - Weighted Cosine (soma ponderada)
   - Reciprocal Rank Fusion (RRF)
   - DAT Linear (alpha adaptativo)
   - Weight Policies (static, heuristic, LLM-based)

5. **Evaluation** (`src/eval/`): Métricas de avaliação
   - nDCG@k (Normalized Discounted Cumulative Gain)
   - MRR (Mean Reciprocal Rank)
   - MAP (Mean Average Precision)
   - Recall@k e Precision@k

6. **Datasets** (`src/datasets/`): Loaders para datasets BEIR
   - SciFact (verificação científica)
   - FIQA (Q&A financeiro)
   - NFCorpus (nutrição/medicina)
   - SQuAD (question answering)

---

## 🚀 Quick Start

### 1. Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/hybrid-retrieval.git
cd hybrid-retrieval

# Setup automático (cria venv e instala dependências)
./setup_venv.sh

# Ou manual
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Download dos Datasets

```bash
# Baixar datasets BEIR (SciFact, FIQA, NFCorpus)
python download_datasets.py \
  --datasets scifact,fiqa,nfcorpus \
  --root ./data \
  --format parquet

# Apenas SciFact (para testes rápidos)
python download_datasets.py \
  --datasets scifact \
  --root ./data \
  --format parquet
```

### 3. Executar Primeiro Experimento

```bash
# Ativar ambiente
source .venv/bin/activate

# Executar experimento baseline (Dense retriever em SciFact)
python scripts/run_experiment.py \
  --config configs/individuals/dense_scifact.yaml

# Ver resultados
cat outputs/experiments/default/dense_scifact.csv
```

### 4. Experimento Tri-Modal

```bash
# Retriever tri-modal com reranking adaptativo
python scripts/run_experiment.py \
  --config configs/trimodal/tri_modal_scifact.yaml

# Comparar com baseline
python scripts/run_experiment.py \
  --config configs/individuals/dense_scifact.yaml

# Resultados em: outputs/experiments/
```

---

## 📁 Estrutura do Projeto

```
hybrid-retrieval/
├── configs/                    # Configurações YAML de experimentos
│   ├── individuals/           # Retrievers individuais (dense, tfidf, bm25, graph)
│   ├── bimodal/               # Combinações bimodais
│   ├── trimodal/              # Tri-modal (semantic+tfidf+graph)
│   ├── dat_experiments/       # Experimentos DAT (Dynamic Alpha Tuning)
│   └── dat_hs/                # Experimentos em dataset HS (HotpotQA-based)
├── data/                      # Datasets BEIR processados
│   ├── scifact/
│   ├── fiqa/
│   ├── nfcorpus/
│   └── squad*/                # SQuAD e variações
├── scripts/                   # Scripts de execução e análise
│   ├── run_experiment.py      # Executor principal (via YAML)
│   ├── run_individual_retrievers.py
│   ├── run_bimodal_benchmark.py
│   ├── evaluate.py            # Avaliação de predições
│   ├── compare_hs_results.py
│   └── ...
├── src/                       # Código-fonte do framework
│   ├── config/                # Sistema de configuração (Pydantic schemas)
│   ├── core/                  # Interfaces (ABCs)
│   ├── datasets/              # Loaders BEIR
│   ├── encoders/              # Building blocks (semantic, tfidf, entities)
│   ├── vectorizers/           # Combinadores de encoders
│   ├── indexes/               # FAISS e outros índices
│   ├── retrievers/            # Retrievers completos
│   ├── fusion/                # Estratégias de fusão e reranking
│   ├── eval/                  # Métricas de avaliação
│   ├── experiments/           # Experiment runner
│   └── utils/                 # Logging e utilitários
├── tests/                     # Testes unitários e de integração
├── outputs/                   # Resultados de experimentos
│   ├── experiments/           # CSVs e JSONs de métricas
│   └── artifacts/             # Índices FAISS, caches de entidades
├── requirements.txt           # Dependências Python
├── setup_venv.sh              # Setup automático
└── README.md                  # Este arquivo
```

---

## 🎯 Como Usar

### Opção 1: Via Configuração YAML (Recomendado)

```yaml
# configs/my_experiment.yaml
experiment:
  name: "my_first_experiment"
  output_dir: "./outputs/experiments/my_first"

dataset:
  name: "scifact"
  root: "./data/scifact/processed/beir"

retrievers:
  - name: "dense_minilm"
    type: "dense"
    vectorizer:
      type: "dense"
      semantic:
        model: "sentence-transformers/all-MiniLM-L6-v2"
    index:
      type: "faiss"
      metric: "ip"

  - name: "trimodal_hybrid"
    type: "hybrid"
    vectorizer:
      type: "tri_modal"
      semantic:
        model: "sentence-transformers/all-MiniLM-L6-v2"
      tfidf:
        dim: 1000
      graph:
        model: "BAAI/bge-large-en-v1.5"
    fusion:
      strategy: "weighted_cosine"
      policy: "heuristic"
    reranker:
      type: "tri_modal"
      topk_first: 150

metrics: ["nDCG", "MRR", "MAP", "Recall", "Precision"]
ks: [1, 3, 5, 10]
output_formats: ["csv", "json"]
```

```bash
python scripts/run_experiment.py --config configs/my_experiment.yaml
```

### Opção 2: Via Python

```python
from pathlib import Path
from src.config.loader import load_config
from src.experiments.runner import ExperimentRunner

# Carregar configuração
config = load_config("configs/my_experiment.yaml")

# Executar experimento
runner = ExperimentRunner(config)
results_df = runner.run()

# Analisar resultados
print(results_df[results_df["k"] == 10][["retriever", "nDCG", "MRR"]])
```

### Opção 3: Uso Programático (Para Desenvolvimento)

```python
from src.retrievers.hybrid_faiss import HybridRetriever
from src.vectorizers.tri_modal_vectorizer import TriModalVectorizer
from src.indexes.hybrid_index import HybridIndex
from src.datasets.loader import load_beir_dataset, as_documents, as_queries

# Carregar dataset
corpus, queries, qrels = load_beir_dataset(Path("data/scifact/processed/beir"))
docs = as_documents(corpus)
qs = as_queries(queries)

# Criar retriever
vectorizer = TriModalVectorizer(
    semantic_model_name="sentence-transformers/all-MiniLM-L6-v2",
    tfidf_dim=1000,
    graph_model_name="BAAI/bge-large-en-v1.5",
)
index = HybridIndex(vectorizer=vectorizer)
retriever = HybridRetriever(vectorizer, index, ...)

# Construir índice
retriever.build_index(docs)

# Buscar
results = retriever.retrieve(qs, k=10)
# results: {"q1": [("doc1", 0.95), ...], ...}
```

---

## 🧪 Experimentos Disponíveis

### 1. Retrievers Individuais (`configs/individuals/`)

Teste cada modalidade separadamente:

```bash
# Dense (semantic apenas)
python scripts/run_experiment.py --config configs/individuals/dense_scifact.yaml

# TF-IDF (lexical apenas)
python scripts/run_experiment.py --config configs/individuals/tfidf_scifact.yaml

# BM25 (baseline lexical)
python scripts/run_experiment.py --config configs/individuals/bm25_scifact.yaml

# Graph (entidades apenas)
python scripts/run_experiment.py --config configs/individuals/graph_scifact.yaml
```

### 2. Combinações Bimodais (`configs/bimodal/`)

```bash
# Semantic + TF-IDF
python scripts/run_experiment.py --config configs/bimodal/semantic_tfidf_scifact.yaml

# Semantic + Graph
python scripts/run_experiment.py --config configs/bimodal/semantic_graph_scifact.yaml

# TF-IDF + Graph
python scripts/run_experiment.py --config configs/bimodal/tfidf_graph_scifact.yaml
```

### 3. Tri-Modal (`configs/trimodal/`)

```bash
# Tri-modal com pesos adaptativos (heurísticos)
python scripts/run_experiment.py --config configs/trimodal/tri_modal_scifact.yaml

# Tri-modal com pesos estáticos
python scripts/run_experiment.py --config configs/trimodal/tri_modal_static_scifact.yaml

# Tri-modal com RRF (Reciprocal Rank Fusion)
python scripts/run_experiment.py --config configs/trimodal/tri_modal_rrf_scifact.yaml
```

### 4. DAT (Dynamic Alpha Tuning) (`configs/dat_experiments/`)

Experimentos usando LLM judges para determinar pesos adaptativos:

```bash
# Baselines (alpha fixo)
python scripts/run_experiment.py --config configs/dat_experiments/01_baseline_alpha_0.0_bm25_only.yaml
python scripts/run_experiment.py --config configs/dat_experiments/03_baseline_alpha_0.6.yaml
python scripts/run_experiment.py --config configs/dat_experiments/04_baseline_alpha_1.0_dense_only.yaml

# DAT com OpenAI
python scripts/run_experiment.py --config configs/dat_experiments/05_dat_hybrid_gpt4o_mini.yaml
python scripts/run_experiment.py --config configs/dat_experiments/06_dat_hybrid_gpt4o.yaml

# DAT com Ollama (local)
python scripts/run_experiment.py --config configs/dat_experiments/09_dat_hybrid_llama31.yaml
python scripts/run_experiment.py --config configs/dat_experiments/10_dat_hybrid_llama_finetune.yaml
```

---

## ⚙️ Configurações Avançadas

### Cache e Artefatos

O framework usa cache agressivo para acelerar experimentos:

```bash
# Estrutura de cache
outputs/
├── artifacts/
│   ├── faiss_indexes/      # Índices FAISS construídos
│   ├── entity_caches/      # Entidades extraídas (NER)
│   └── llm_judge_cache/    # Respostas de LLM judges
```

Para forçar rebuild:

```yaml
# No YAML de configuração
index:
  type: "faiss"
  force_rebuild: true  # Força reconstrução do índice

vectorizer:
  graph:
    entity_cache: "./cache/entities"
    force_rebuild: true  # Força re-extração de entidades
```

### Otimizações para Máquinas com Pouca Memória

```yaml
# Reduzir batch size de NER
vectorizer:
  graph:
    ner_batch_size: 4      # Padrão: 8
    ner_n_process: 1       # Padrão: 4

# Usar índice FAISS comprimido
index:
  type: "faiss"
  factory: "IVF4096,Flat"  # Ao invés de FlatIP (exato)
  nprobe: 64
```

### GPU Acceleration

```yaml
# FAISS automaticamente detecta e usa GPU se disponível
index:
  type: "faiss"
  use_gpu: true  # Tenta usar GPU (MPS no Mac M1, CUDA no NVIDIA)
```

---

## 📊 Avaliação e Métricas

### Métricas Disponíveis

- **nDCG@k**: Normalized Discounted Cumulative Gain (principal métrica)
- **MRR**: Mean Reciprocal Rank (ranking do primeiro relevante)
- **MAP**: Mean Average Precision (precisão média)
- **Recall@k**: Fração de documentos relevantes recuperados
- **Precision@k**: Fração de documentos recuperados que são relevantes

### Análise de Resultados

```python
import pandas as pd

# Carregar resultados
df = pd.read_csv("outputs/experiments/my_experiment/results.csv")

# Comparar retrievers em nDCG@10
summary = df[df["k"] == 10].groupby("retriever")["nDCG"].mean()
print(summary.sort_values(ascending=False))

# Visualizar performance por k
import matplotlib.pyplot as plt

for retriever in df["retriever"].unique():
    subset = df[df["retriever"] == retriever]
    plt.plot(subset["k"], subset["nDCG"], label=retriever, marker='o')

plt.xlabel("k")
plt.ylabel("nDCG@k")
plt.legend()
plt.title("Performance por Retriever")
plt.show()
```

---

## 🛠️ Extensibilidade

O framework foi projetado para ser facilmente extensível. Para adicionar novos componentes:

### Adicionar Novo Vectorizer

1. Crie classe implementando `AbstractVectorizer` em `src/vectorizers/`
2. Registre na factory `src/vectorizers/factory.py`
3. Atualize schema em `src/config/schema.py`
4. Use via YAML

Exemplo: veja `src/vectorizers/README.md`

### Adicionar Novo Retriever

1. Crie classe implementando `AbstractRetriever` em `src/retrievers/`
2. Registre na factory `src/retrievers/factory.py`
3. Atualize schema
4. Use via YAML

Exemplo: veja `src/retrievers/README.md`

### Adicionar Nova Estratégia de Fusão

1. Crie classe implementando `AbstractFusionStrategy` em `src/fusion/strategies.py`
2. Registre no dict `FUSION_STRATEGIES`
3. Use via YAML

Exemplo: veja `src/fusion/README.md`

### Adicionar Nova Métrica

1. Crie classe implementando `AbstractMetric` em `src/eval/metrics.py`
2. Registre em `METRICS_REGISTRY`
3. Use via YAML em `metrics: ["MinhaMetrica"]`

---

## 📚 Datasets Suportados

### SciFact
- **Domínio**: Verificação científica (biomedicina)
- **Tamanho**: ~5K docs, ~300 queries
- **Uso**: Fact-checking de claims científicas

### FIQA
- **Domínio**: Finanças (Q&A de fóruns)
- **Tamanho**: ~57K docs, ~6.6K queries
- **Uso**: Question answering financeiro

### NFCorpus
- **Domínio**: Nutrição e medicina
- **Tamanho**: ~3.6K docs, ~323 queries
- **Uso**: Linking entre artigos científicos e divulgação

### SQuAD (Variações Customizadas)
- **squad_small**: Subset pequeno para testes rápidos
- **squad_llm_judge**: Dataset com julgamentos de LLM para treinar judges

---

## 🧩 Dependências Principais

- **sentence-transformers**: Embeddings semânticos
- **transformers**: Modelos Hugging Face
- **faiss-cpu** (ou faiss-gpu): Busca vetorial rápida
- **scikit-learn**: TF-IDF e métricas
- **scispacy**: NER para domínio científico
- **rank-bm25**: Implementação BM25
- **pydantic**: Validação de configurações
- **pandas**, **numpy**, **pyarrow**: Manipulação de dados

Veja `requirements.txt` para lista completa.

---

## 🔬 Aplicações

Este framework é útil para:

- ✅ **Pesquisa em RAG**: Testar diferentes estratégias de retrieval
- ✅ **Benchmarking**: Comparar métodos em datasets padronizados
- ✅ **Produção**: Base para sistemas de busca híbrida
- ✅ **Educação**: Aprender sobre IR e RAG na prática
- ✅ **Desenvolvimento**: Prototipar novas técnicas de retrieval

---

## 📖 Documentação Adicional

Para documentação detalhada de cada módulo:

- `src/README.md` - Visão geral da arquitetura
- `src/vectorizers/README.md` - Como criar vectorizers
- `src/retrievers/README.md` - Como criar retrievers
- `src/fusion/README.md` - Estratégias de fusão e reranking
- `src/eval/README.md` - Métricas de avaliação
- `src/datasets/README.md` - Datasets BEIR
- `configs/README.md` - Estrutura de configurações

---

## 🧪 Testes

```bash
# Executar todos os testes
pytest

# Testes de integração
pytest tests/test_integration_modular.py

# Testes de métricas
pytest tests/test_metrics_*.py

# Testes end-to-end
pytest tests/test_cli_end2end_scifact.py
```

---

## 📚 Referências

### Papers

1. **BEIR**: Thakur, N., et al. (2021). "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models"
2. **DAT**: "Rethinking Hybrid Retrieval: When Small Embeddings and LLM Re-ranking Beat Bigger Models" (2024)
3. **RRF**: Cormack, G. V., et al. (2009). "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
4. **BM25**: Robertson, S., Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond"

### Datasets

- **SciFact**: Wadden, D., et al. (2020). "Fact or Fiction: Verifying Scientific Claims"
- **FIQA**: Maia, M., et al. (2018). "WWW'18 Open Challenge: Financial Opinion Mining and Question Answering"
- **NFCorpus**: Boteva, V., et al. (2016). "A Full-Text Learning to Rank Dataset for Medical Information Retrieval"

### Bibliotecas

- **FAISS**: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- **Sentence-Transformers**: [https://www.sbert.net/](https://www.sbert.net/)
- **spaCy**: [https://spacy.io/](https://spacy.io/)
- **scikit-learn**: [https://scikit-learn.org/](https://scikit-learn.org/)
