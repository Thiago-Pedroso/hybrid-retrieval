# 🔀 Fusion

Este módulo contém **estratégias de fusão, políticas de pesos e rerankers** que combinam e refinam resultados de múltiplas modalidades ou métodos de retrieval.

---

## 📋 **Visão Geral**

O módulo `fusion/` é responsável por:

1. **Fusion Strategies**: Combinar múltiplas listas ranqueadas em uma única
2. **Weight Policies**: Determinar pesos adaptativos para cada modalidade
3. **Rerankers**: Refinar ranking de candidatos usando recálculo de scores
4. **Normalization**: Normalizar scores de diferentes métodos

**Fluxo típico**:
```
Múltiplas Listas Ranqueadas → Fusion Strategy → Lista Única
Query → Weight Policy → Pesos Adaptativos
Candidatos + Pesos → Reranker → Ranking Refinado
```

---

## 🎯 **Componentes Disponíveis**

### **1. Fusion Strategies** (`strategies.py`)

Estratégias para combinar múltiplas listas ranqueadas.

#### **WeightedCosineFusion**

Combina resultados por soma ponderada de scores.

```python
from src.fusion.strategies import WeightedCosineFusion

fusion = WeightedCosineFusion()

results_list = [
    {"q1": [("doc1", 0.9), ("doc2", 0.8)]},  # Lista 1
    {"q1": [("doc2", 0.85), ("doc1", 0.75)]},  # Lista 2
]

weights = [0.6, 0.4]  # 60% lista 1, 40% lista 2
fused = fusion.fuse("query text", results_list, weights=weights)
# Retorna: [("doc1", 0.84), ("doc2", 0.82)]
# doc1: 0.9*0.6 + 0.75*0.4 = 0.84
# doc2: 0.8*0.6 + 0.85*0.4 = 0.82
```

**Uso**: Quando você quer combinar scores diretamente.

---

#### **ReciprocalRankFusion (RRF)**

Combina resultados usando RRF: `score = sum(1 / (k + rank))`.

```python
from src.fusion.strategies import ReciprocalRankFusion

fusion = ReciprocalRankFusion(k=60)  # k padrão = 60

results_list = [
    {"q1": [("doc1", 0.9), ("doc2", 0.8)]},
    {"q1": [("doc2", 0.85), ("doc1", 0.75)]},
]

fused = fusion.fuse("query text", results_list)
# doc1: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
# doc2: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
# (Empate, mas doc1 aparece primeiro na lista 1, então pode ter prioridade)
```

**Uso**: Quando você quer combinar rankings sem depender de scores absolutos.

---

#### **DATLinearFusion**

Combinação linear com alpha adaptativo: `R(q,d) = α * S̃_dense + (1-α) * S̃_BM25`.

```python
from src.fusion.strategies import DATLinearFusion

fusion = DATLinearFusion(alpha=0.6)  # 60% dense, 40% BM25

results_list = [
    {"q1": [("doc1", 0.9), ("doc2", 0.8)]},  # BM25
    {"q1": [("doc2", 0.85), ("doc1", 0.75)]},  # Dense
]

fused = fusion.fuse("query text", results_list)
# doc1: 0.6*0.75 + 0.4*0.9 = 0.81
# doc2: 0.6*0.85 + 0.4*0.8 = 0.83
```

**Uso**: Para DAT (Dynamic Alpha Tuning) com alpha fixo ou adaptativo.

---

### **2. Weight Policies** (`weighting.py`)

Políticas que determinam pesos adaptativos baseado na query.

#### **StaticPolicy**

Pesos fixos (não adaptativos).

```python
from src.fusion.weighting import StaticPolicy

policy = StaticPolicy(ws=0.5, wt=0.3, wg=0.2)  # semantic, tfidf, graph

weights = policy.weights("any query")
# Retorna: (0.5, 0.3, 0.2)  # Sempre o mesmo
```

**Uso**: Quando você quer pesos fixos (baseline).

---

#### **HeuristicLLMPolicy**

Pesos adaptativos baseados em heurísticas da query.

```python
from src.fusion.weighting import HeuristicLLMPolicy

policy = HeuristicLLMPolicy()

weights = policy.weights("What is 42?")
# Query curta com números → mais peso em TF-IDF
# Retorna: (0.4, 0.5, 0.1)  # Ajustado para TF-IDF

weights = policy.weights("Machine learning and artificial intelligence")
# Query longa → mais peso em semantic
# Retorna: (0.5, 0.3, 0.2)  # Ajustado para semantic
```

**Uso**: Quando você quer pesos adaptativos simples (sem LLM).

---

#### **DATWeightPolicy**

Pesos baseados em LLM judge (DAT).

```python
from src.fusion.weighting import DATWeightPolicy

policy = DATWeightPolicy()

# Alpha é calculado via judge scores (não via query text)
alpha = policy.compute_alpha(e_dense=4, e_bm25=3)
# Retorna: 0.6  # 4/5 / (4/5 + 3/5) = 0.57 ≈ 0.6
```

**Uso**: Para DAT com LLM judge.

---

### **3. Rerankers** (`reranker.py`)

Rerankers que refinam ranking de candidatos.

#### **TriModalReranker**

Reranking tri-modal com weighted cosine.

```python
from src.fusion.reranker import TriModalReranker
from src.vectorizers.tri_modal_vectorizer import TriModalVectorizer

vectorizer = TriModalVectorizer(...)
reranker = TriModalReranker(vectorizer)

candidates = [("doc1", "text1"), ("doc2", "text2")]
weights = (0.5, 0.3, 0.2)  # semantic, tfidf, graph

reranked = reranker.rescore("query text", candidates, weights=weights)
# Retorna: [("doc1", 0.92), ("doc2", 0.87)]
# Recalcula scores usando weighted cosine por fatia
```

**Uso**: Para reranking tri-modal com pesos adaptativos.

---

#### **BiModalReranker**

Reranking bi-modal (semantic + TF-IDF).

```python
from src.fusion.reranker import BiModalReranker
from src.vectorizers.bi_modal_vectorizer import BiModalVectorizer

vectorizer = BiModalVectorizer(...)
reranker = BiModalReranker(vectorizer)

reranked = reranker.rescore("query text", candidates, weights=(0.6, 0.4))
```

**Uso**: Para reranking bi-modal.

---

### **4. LLM Judge** (`llm_judge.py`)

LLM judge para DAT (avalia efetividade de métodos).

```python
from src.fusion.llm_judge import LLMJudge

judge = LLMJudge(
    model="gpt-4o-mini",
    temperature=0.0,
    cache_dir="./cache/llm_judge",
)

# Avalia efetividade de BM25 e Dense para uma query
e_bm25, e_dense = judge.judge_query(
    query="What is machine learning?",
    bm25_results=[("doc1", 0.9), ("doc2", 0.8)],
    dense_results=[("doc3", 0.85), ("doc1", 0.75)],
)
# Retorna: (3, 4)  # BM25=3/5, Dense=4/5
```

**Uso**: Para DAT com alpha adaptativo baseado em LLM.

---

## ➕ **Como Adicionar Nova Estratégia de Fusão**

### **Passo 1: Criar Classe que Implementa AbstractFusionStrategy**

```python
# src/fusion/strategies.py

class BordaCountFusion(AbstractFusionStrategy):
    """Borda Count: soma posições inversas"""
    
    def fuse(
        self,
        query: str,
        results_list: List[Dict[str, List[Tuple[str, float]]]],
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        if not results_list:
            return []
        
        if weights is None:
            weights = [1.0] * len(results_list)
        
        # Borda Count: score = soma de (n - rank + 1) para cada lista
        doc_scores: Dict[str, float] = {}
        n = max(len(r[next(iter(r.keys()))]) for r in results_list if r)
        
        for result_dict, weight in zip(results_list, weights):
            query_id = next(iter(result_dict.keys()))
            for rank, (doc_id, _) in enumerate(result_dict[query_id], start=1):
                borda_score = (n - rank + 1) * weight
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + borda_score
        
        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

# Registrar
FUSION_STRATEGIES["borda_count"] = BordaCountFusion
```

### **Passo 2: Atualizar Schema**

```python
# src/config/schema.py

class FusionConfig(BaseModel):
    strategy: Literal["weighted_cosine", "reciprocal_rank", "dat_linear", "borda_count"]
    # ...
```

---

## ➕ **Como Adicionar Nova Weight Policy**

### **Passo 1: Criar Classe que Implementa AbstractWeightPolicy**

```python
# src/fusion/weighting.py

class QueryLengthPolicy(AbstractWeightPolicy):
    """Ajusta pesos baseado no comprimento da query"""
    
    def weights(self, query_text: str) -> Tuple[float, ...]:
        query_len = len(query_text.split())
        
        if query_len <= 3:
            return (0.3, 0.5, 0.2)  # Query curta: mais TF-IDF
        elif query_len <= 10:
            return (0.4, 0.4, 0.2)  # Query média: balanceado
        else:
            return (0.5, 0.3, 0.2)  # Query longa: mais semantic
```

### **Passo 2: Registrar na Factory**

```python
# src/fusion/factory.py

elif policy_type == "query_length":
    return QueryLengthPolicy()
```

---

## ➕ **Como Adicionar Novo Reranker**

### **Passo 1: Criar Classe que Implementa AbstractReranker**

```python
# src/fusion/reranker.py

class CrossEncoderReranker(AbstractReranker):
    """Reranker usando Cross-Encoder (BERT fine-tuned)"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
    
    def rescore(
        self,
        query_text: str,
        candidate_docs: List[Tuple[str, str]],
        weights: Optional[Tuple[float, ...]] = None,
    ) -> List[Tuple[str, float]]:
        # Cross-Encoder recebe pares (query, doc)
        pairs = [[query_text, doc_text] for _, doc_text in candidate_docs]
        
        # Predizer scores
        scores = self.model.predict(pairs)
        
        # Combinar com doc_ids
        results = [
            (doc_id, float(score))
            for (doc_id, _), score in zip(candidate_docs, scores)
        ]
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
```

---

## ✅ **Boas Práticas**

### **1. Normalize Pesos Quando Apropriado**

```python
# ✅ BOM - Normaliza pesos
def weights(self, query_text: str) -> Tuple[float, ...]:
    ws, wt, wg = 0.5, 0.3, 0.2
    total = ws + wt + wg
    return (ws/total, wt/total, wg/total)

# ❌ RUIM - Pesos não normalizados
def weights(self, query_text: str) -> Tuple[float, ...]:
    return (0.5, 0.3, 0.2)  # Total = 1.0, mas e se mudar?
```

### **2. Trate Listas Vazias em Fusion**

```python
# ✅ BOM - Trata lista vazia
def fuse(self, query: str, results_list: List[...], weights: Optional[List[float]] = None):
    if not results_list:
        return []
    # ... fusão normal ...

# ❌ RUIM - Pode quebrar com lista vazia
def fuse(self, query: str, results_list: List[...], weights: Optional[List[float]] = None):
    return self._combine(results_list)  # Pode dar erro se vazio!
```

### **3. Use Cache para LLM Judge (quando aplicável)**

```python
# ✅ BOM - Cache de judge
judge = LLMJudge(cache_dir="./cache/llm_judge")
# Reutiliza resultados para queries similares

# ❌ RUIM - Sem cache (chamadas repetidas)
judge = LLMJudge()  # Sem cache, sempre chama API
```

### **4. Documente Fórmulas de Fusão**

```python
class MyFusion(AbstractFusionStrategy):
    """
    Minha estratégia de fusão.
    
    Fórmula: score(doc) = sum(weight_i * score_i(doc)) / sum(weights)
    """
    # ...
```