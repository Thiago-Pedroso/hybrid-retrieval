# 🔬 Hybrid Retrieval: Tri-Modal Fusion

Autor: Thiago Pedroso de Jesus

Repositório destinado ao desenvolvimento do projeto final da Residência em IA no Bacharelado em Inteligência 
Artificial da UFG.

**Replicação do paper**: ["Rethinking Hybrid Retrieval: When Small Embeddings and LLM Re-ranking Beat Bigger Models"](https://arxiv.org/pdf/2506.00049)

O objetivo é fazer um sistema de **retrieval híbrido tri-modal** que combina embeddings semânticos, representações lexicais (TF-IDF) e embeddings de entidades para recuperação de documentos.

## 🎯 **Objetivo do Paper**

O paper ["Rethinking Hybrid Retrieval"](https://arxiv.org/pdf/2506.00049) demonstra que:

1. **Embeddings pequenos** (MiniLM) podem superar modelos maiores (BGE-Large) em retrieval híbrido
2. **LLM re-ranking** na segunda fase é mais importante que embeddings grandes
3. **Tri-modal fusion** (semantic + lexical + graph) é superior a abordagens unimodais
4. **Pesos adaptativos** são cruciais para diferentes tipos de queries

---

## 🏗️ **Arquitetura do Sistema**

### **Tri-Modal Vectorizer**

O sistema combina três modalidades:

1. **Semantic (s)**: Embeddings de sentenças
   - MiniLM-L6-v2: 384d
   - BGE-Large: 1024d

2. **Lexical (t)**: TF-IDF sparse
   - Dimensão: 1000d
   - Backend: scikit-learn

3. **Graph (g)**: Embeddings de entidades
   - Extração: spaCy/scispaCy NER
   - Embedding: BGE-Large (1024d)
   - Agregação: TF-IDF weighted

### **Pipeline de Retrieval**

```
Query → [s; t; g] → FAISS Search → Re-ranking → Top-k Results
```

1. **Encoding**: Query convertida para vetor tri-modal `[semantic; tfidf; entities]`
2. **Search**: Busca por similaridade no índice FAISS
3. **Re-ranking**: Combinação de cosenos com pesos adaptativos
4. **Output**: Top-k documentos ranqueados

---

## 📈 **Métricas Utilizadas**

### **Métricas de Retrieval**

- **nDCG@k**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank  
- **MAP**: Mean Average Precision
- **Recall@k**: Taxa de recuperação
- **Precision@k**: Precisão dos top-k

### **Métricas de Performance**

- **Tempo de Indexação**: Build do índice FAISS
- **Tempo de Retrieval**: Busca + re-ranking
- **Uso de Memória**: RAM peak durante processamento

---

## 🗂️ **Datasets Suportados**

### **BEIR Benchmark**

- **SciFact**: Scientific Claims
- **FIQA**: Financial Q&A dataset
- **NFCorpus**: Medical domain corpus

### **Formato de Dados**

```
data/
├── scifact/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels/
│       ├── test.jsonl
│       └── dev.jsonl
```

---

## 📁 **Estrutura do Projeto**

```
hybrid-retrieval/
├── src/
│   ├── datasets/          # Loaders de datasets BEIR
│   ├── retrievers/        # Implementações de retrieval
│   ├── tri_modal/         # Vectorizer tri-modal
│   ├── eval/             # Métricas de avaliação
│   └── utils/             # Logging e utilitários
├── scripts/
│   ├── run_experiment.py  # Script principal para experimentos via YAML
│   ├── run_prerank_ab.py  # Script especializado A/B test
│   └── evaluate.py        # Avaliação de predições pré-computadas
├── tests/                 # Testes unitários
├── docs/                  # Documentação técnica
├── outputs/               # Resultados e caches
└── requirements.txt       # Dependências
```

---

## 🚀 **Como Usar**

### **Setup do Ambiente**

```bash
# Clone o repositório
git clone <repo-url>
cd hybrid-retrieval

# Setup automático
./setup_venv.sh

# Ou manual
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### **Execução Básica**

```bash
# Ativar ambiente
source .venv/bin/activate

# Teste SciFact
python scripts/run_prerank_ab.py \
  --dataset scifact \
  --k 10

# A/B Test completo (MiniLM vs BGE-Large)
python scripts/run_prerank_ab.py \
  --dataset scifact \
  --k 10 \
  --semantic-a "sentence-transformers/all-MiniLM-L6-v2" \
  --semantic-b "BAAI/bge-large-en-v1.5" \
  --graph-model "BAAI/bge-large-en-v1.5" \
  --csv-out ./outputs/scifact_results.csv
```

### **Todos os Datasets**

```bash
python scripts/run_prerank_ab.py \
  --all \
  --k 10 \
  --csv-out ./outputs/all_results.csv
```

---

## ⚙️ **Configurações Avançadas**

### **Otimizações para Mac M1 8GB**

```bash
# Reduzir batch size se RAM > 90%
python scripts/run_prerank_ab.py \
  --dataset scifact \
  --ner-batch-size 4 \
  --ner-n-process 1

# Usar GPU (MPS) se disponível
python scripts/run_prerank_ab.py \
  --dataset scifact \
  --device mps
```

### **Parâmetros de FAISS**

```bash
# Índice otimizado para velocidade
python scripts/run_prerank_ab.py \
  --dataset scifact \
  --faiss-factory "OPQ64,IVF4096,PQ64x8" \
  --faiss-nprobe 64
```

### **Cache e Rebuild**

```bash
# Limpar cache de entidades
python scripts/run_prerank_ab.py \
  --dataset scifact \
  --entity-force-rebuild

# Limpar cache de índice
python scripts/run_prerank_ab.py \
  --dataset scifact \
  --index-force-rebuild
```

---

## 📊 **Resultados Iniciais Obtidos**

### **SciFact Dataset (nDCG@10)**

| Modelo | nDCG@10 | MRR | MAP | Recall | Precision |
|--------|---------|-----|-----|--------|-----------|
| **MiniLM-L6-v2** | **0.552** ✅ | 0.505 | 0.494 | 0.721 | 0.081 |
| **BGE-Large** | 0.414 | 0.370 | 0.355 | 0.584 | 0.065 |

**Observação**: MiniLM-L6-v2 superou BGE-Large, possivelmente devido a:
- Pesos não otimizados para BGE-Large (redundância semantic + entities)
- Noun chunks gerando ruído nas entidades

---