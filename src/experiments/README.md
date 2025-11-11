# 🧪 Experiments

Este módulo contém o **ExperimentRunner** que executa experimentos completos de retrieval a partir de configurações YAML/JSON.

---

## 📋 **Visão Geral**

O `ExperimentRunner` orquestra o pipeline completo:

1. **Carrega configuração**: YAML/JSON → `ExperimentConfig` validado
2. **Carrega datasets**: BEIR format → corpus, queries, qrels
3. **Executa retrievers**: Para cada retriever, constrói índice e busca
4. **Avalia resultados**: Calcula métricas (nDCG, MRR, MAP, etc.)
5. **Salva resultados**: CSV, JSON, JSONL

**Fluxo**:
```
Config YAML → ExperimentRunner → Resultados CSV/JSON
```

---

## 🎯 **Componentes**

### **ExperimentRunner**

Runner principal que executa experimentos.

```python
from src.experiments.runner import ExperimentRunner
from src.config.loader import load_config

# Carregar configuração
config = load_config("configs/my_experiment.yaml")

# Criar runner
runner = ExperimentRunner(config)

# Executar experimento
results_df = runner.run()
# Retorna: DataFrame com métricas por retriever, dataset, k
```

**Output do DataFrame**:
- Colunas: `k`, `retriever`, `retriever_type`, `dataset`, `split`, `nDCG`, `MRR`, `MAP`, `Recall`, `Precision`, `t_retrieve_sec`
- Uma linha por combinação (retriever × dataset × k)

---

### **run_experiment()**

Função helper para executar experimento diretamente.

```python
from src.experiments.runner import run_experiment

# Executar experimento
results_df = run_experiment("configs/my_experiment.yaml")

# Salvar manualmente (se necessário)
results_df.to_csv("outputs/results.csv", index=False)
```

---

## 📝 **Exemplo de Uso**

### **Configuração YAML**

```yaml
# configs/my_experiment.yaml
experiment:
  name: "my_experiment"
  output_dir: "./outputs/experiments/my_experiment"

dataset:
  name: "scifact"
  root: "./data/scifact/processed/beir"
  split_preference: ["test", "dev", "validation", "train"]

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

  - name: "trimodal_heuristic"
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
    index:
      type: "faiss"
      metric: "ip"

metrics:
  - "nDCG"
  - "MRR"
  - "MAP"
  - "Recall"
  - "Precision"

ks: [1, 3, 5, 10]

output_formats:
  - "csv"
  - "json"
```

### **Executar Experimento**

```python
from src.experiments.runner import run_experiment

# Executar
results_df = run_experiment("configs/my_experiment.yaml")

# Ver resultados
print(results_df.head())

# Filtrar por retriever
dense_results = results_df[results_df["retriever"] == "dense_minilm"]
print(dense_results[["k", "nDCG", "MRR"]])
```

### **Múltiplos Datasets**

```yaml
# configs/multi_dataset.yaml
experiment:
  name: "multi_dataset"
  output_dir: "./outputs/experiments/multi_dataset"

datasets:
  - name: "scifact"
    root: "./data/scifact/processed/beir"
  - name: "fiqa"
    root: "./data/fiqa/processed/beir"
  - name: "nfcorpus"
    root: "./data/nfcorpus/processed/beir"

retrievers:
  - name: "trimodal"
    type: "hybrid"
    # ... configuração ...

metrics: ["nDCG", "MRR"]
ks: [1, 5, 10]
```

---

## ✅ **Boas Práticas**

### **1. Use Nomes Descritivos para Retrievers**

```yaml
# ✅ BOM - Nome descritivo
retrievers:
  - name: "dense_minilm_l6_v2"
    type: "dense"
    # ...

# ❌ RUIM - Nome genérico
retrievers:
  - name: "retriever1"
    type: "dense"
    # ...
```

### **2. Organize Outputs por Experimento**

```yaml
# ✅ BOM - Output organizado
experiment:
  name: "scifact_baseline_vs_hybrid"
  output_dir: "./outputs/experiments/scifact_comparison"

# ❌ RUIM - Output genérico
experiment:
  name: "test"
  output_dir: "./outputs"
```

### **3. Use Split Preference Apropriada**

```yaml
# ✅ BOM - Preferência clara
dataset:
  split_preference: ["test", "dev", "validation", "train"]
  # Tenta test primeiro, depois dev, etc.

# ❌ RUIM - Sem preferência (pode usar split errado)
dataset:
  # Sem split_preference
```

### **4. Inclua Métricas Relevantes**

```yaml
# ✅ BOM - Métricas completas
metrics:
  - "nDCG"  # Principal
  - "MRR"   # Ranking
  - "MAP"   # Precisão média
  - "Recall"  # Cobertura
  - "Precision"  # Precisão

# ⚠️ OK - Métricas mínimas (se performance for crítica)
metrics:
  - "nDCG"
  - "MRR"
```

### **5. Teste com k Pequeno Primeiro**

```yaml
# ✅ BOM - Testa com k pequeno primeiro
ks: [1, 3, 5, 10]  # Começa pequeno

# ❌ RUIM - k muito grande logo de cara
ks: [1, 10, 50, 100]  # 100 pode ser lento demais
```

---

## 🔍 **Tratamento de Erros**

O `ExperimentRunner` trata erros graciosamente:

- **Dataset não encontrado**: Adiciona linhas de erro no DataFrame
- **Erro ao carregar dataset**: Loga erro e continua com próximo dataset
- **Erro ao executar retriever**: Loga erro e adiciona linhas de erro no DataFrame

**Exemplo de output com erro**:
```python
# DataFrame inclui coluna "error" quando há problemas
error_rows = results_df[results_df["error"].notna()]
print(error_rows[["retriever", "dataset", "error"]])
```

---

## 📊 **Análise de Resultados**

### **Agregar por Retriever**

```python
import pandas as pd

# Média de nDCG@10 por retriever
summary = results_df[results_df["k"] == 10].groupby("retriever")["nDCG"].mean()
print(summary.sort_values(ascending=False))
```

### **Comparar Retrievers**

```python
# Comparar dois retrievers
ret1 = results_df[(results_df["retriever"] == "dense_minilm") & (results_df["k"] == 10)]
ret2 = results_df[(results_df["retriever"] == "trimodal") & (results_df["k"] == 10)]

print(f"Dense nDCG@10: {ret1['nDCG'].mean():.4f}")
print(f"TriModal nDCG@10: {ret2['nDCG'].mean():.4f}")
```

### **Exportar para Excel**

```python
# Salvar em Excel (se necessário)
results_df.to_excel("outputs/results.xlsx", index=False)
```

---

## 🚀 **Executar via CLI**

```bash
# Via script
python scripts/run_experiment.py --config configs/my_experiment.yaml

# Direto via Python
python -c "from src.experiments.runner import run_experiment; run_experiment('configs/my_experiment.yaml')"
```

---

## 📚 **Referências**

- Veja `src/config/README.md` para configuração
- Veja `src/eval/README.md` para métricas
- Veja `src/datasets/README.md` para datasets

---

**Última atualização**: 2024

