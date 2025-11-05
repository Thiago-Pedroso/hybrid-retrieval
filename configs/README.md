# Configurações de Experimentos

Este diretório contém arquivos YAML de configuração organizados por tipo de execução.

## 📁 Estrutura

```
configs/
├── individuals/     # Retrievers individuais (dense, tfidf, graph, bm25)
├── bimodal/         # Combinações bimodais (semantic+tfidf, semantic+graph, tfidf+graph)
├── trimodal/        # Combinações trimodais (semantic+tfidf+graph)
└── README.md        # Este arquivo
```

## 🎯 Como Usar

### Executar um Experimento

```bash
# Individual
python scripts/run_experiment.py --config configs/individuals/dense_scifact.yaml

# Bimodal
python scripts/run_experiment.py --config configs/bimodal/semantic_tfidf_scifact.yaml

# Trimodal
python scripts/run_experiment.py --config configs/trimodal/tri_modal_scifact.yaml
```

### Habilitar Múltiplos Datasets

Para executar em múltiplos datasets, descomente a seção `datasets:` e comente a seção `dataset:`:

```yaml
# dataset:
#   name: "scifact"
#   ...

datasets:
  - name: "scifact"
    root: "./data/scifact/processed/beir"
  - name: "fiqa"
    root: "./data/fiqa/processed/beir"
```

## 📋 Disponíveis

### Individuals (`individuals/`)

1. **dense_scifact.yaml** - Retriever denso (semantic embeddings apenas)
2. **tfidf_scifact.yaml** - Retriever TF-IDF (lexical apenas)
3. **graph_scifact.yaml** - Retriever graph (entity embeddings apenas)
4. **bm25_scifact.yaml** - Retriever BM25 (baseline lexical)

### Bimodal (`bimodal/`)

1. **semantic_tfidf_scifact.yaml** - Semantic + TF-IDF
2. **semantic_graph_scifact.yaml** - Semantic + Graph
3. **tfidf_graph_scifact.yaml** - TF-IDF + Graph

### Trimodal (`trimodal/`)

1. **tri_modal_scifact.yaml** - Trimodal com política heurística (adaptativa)
2. **tri_modal_static_scifact.yaml** - Trimodal com pesos estáticos (0.33, 0.33, 0.34)
3. **tri_modal_rrf_scifact.yaml** - Trimodal com Reciprocal Rank Fusion (RRF)

## ⚙️ Parâmetros Principais

### Fusion Strategies

- **weighted_cosine**: Combinação ponderada de similaridades cosseno
- **reciprocal_rank_fusion**: Fusão por ranking recíproco (RRF)

### Weight Policies

- **heuristic**: Pesos adaptativos baseados em características da query
- **static**: Pesos fixos especificados em `weights: [w1, w2, w3]`

### Indexes

- **faiss** com `factory: null`: IndexFlatIP (busca exata)
- **faiss** com `factory: "OPQ64,IVF4096,PQ64x8"`: Índice comprimido (mais rápido)

## 📊 Resultados

Os resultados são salvos em `output_dir` especificado em cada configuração:

- **CSV**: `{output_dir}/{experiment_name}.csv`
- **JSON**: `{output_dir}/{experiment_name}.json`

## 🔧 Customização

Para personalizar um experimento:

1. Copie um arquivo YAML existente
2. Modifique os parâmetros desejados:
   - Modelos: `semantic.model`, `graph.model`
   - Dimensões: `tfidf.dim`
   - Pesos de fusão: `fusion.weights`
   - Métricas: `metrics`
   - K values: `ks`
3. Execute com o novo arquivo

