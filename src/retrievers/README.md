# 🔎 Retrievers

Este módulo contém **retrievers** que orquestram o pipeline completo de retrieval: vectorizer → index → fusion → reranker. Retrievers são o ponto de entrada principal para buscar documentos.

---

## 📋 **Visão Geral**

Retrievers coordenam todos os componentes do pipeline:

```
Documents → build_index() → Índice Construído
Query → retrieve() → Top-K Documentos Ranqueados
```

**Fluxo interno de `retrieve()`**:
1. **Vectorizer**: Converte query em vetor
2. **Index**: Busca candidatos (top-K inicial, ex: 150)
3. **Weight Policy**: Calcula pesos adaptativos
4. **Reranker**: Refina ranking dos candidatos
5. **Retorna**: Top-K final

---

## 🎯 **Retrievers Disponíveis**

### **1. DenseFaiss**

Retriever apenas com embeddings densos (semânticos).

```python
from src.retrievers.dense_faiss import DenseFaiss

retriever = DenseFaiss(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
    artifact_dir="./artifacts",
    index_name="dense.index",
)

retriever.build_index(documents)
results = retriever.retrieve(queries, k=10)
# Retorna: {"query_1": [("doc_1", 0.95), ...], "query_2": [...]}
```

**Uso**: Quando você quer apenas similaridade semântica.

---

### **2. TFIDFRetriever**

Retriever apenas com TF-IDF (lexical).

```python
from src.retrievers.tfidf_faiss import TFIDFRetriever

retriever = TFIDFRetriever(
    dim=1000,
    min_df=2,
    backend="sklearn",
    artifact_dir="./artifacts",
)

retriever.build_index(documents)
results = retriever.retrieve(queries, k=10)
```

**Uso**: Quando você quer apenas matching lexical (exato).

---

### **3. BM25Basic**

Retriever BM25 (baseline lexical).

```python
from src.retrievers.bm25_basic import BM25Basic

retriever = BM25Basic(k1=0.9, b=0.4)

retriever.build_index(documents)
results = retriever.retrieve(queries, k=10)
```

**Uso**: Baseline lexical para comparação.

---

### **4. GraphRetriever**

Retriever apenas com embeddings de entidades.

```python
from src.retrievers.graph_faiss import GraphRetriever

retriever = GraphRetriever(
    graph_model_name="BAAI/bge-large-en-v1.5",
    ner_backend="scispacy",
    artifact_dir="./artifacts",
)

retriever.build_index(documents)
results = retriever.retrieve(queries, k=10)
```

**Uso**: Quando você quer apenas representação baseada em entidades.

---

### **5. HybridRetriever**

Retriever tri-modal com reranking (mais completo).

```python
from src.retrievers.hybrid_faiss import HybridRetriever
from src.vectorizers.tri_modal_vectorizer import TriModalVectorizer
from src.indexes.hybrid_index import HybridIndex
from src.fusion.reranker import TriModalReranker
from src.fusion.weighting import HeuristicLLMPolicy

# Criar componentes
vectorizer = TriModalVectorizer(...)
index = HybridIndex(vectorizer=vectorizer, ...)
reranker = TriModalReranker(vectorizer)
policy = HeuristicLLMPolicy()

# Criar retriever
retriever = HybridRetriever(
    vectorizer=vectorizer,
    index=index,
    reranker=reranker,
    weight_policy=policy,
    topk_first=150,  # Top-K antes de reranking
)

retriever.build_index(documents)
results = retriever.retrieve(queries, k=10)
```

**Uso**: Quando você quer máxima qualidade com tri-modal + reranking.

---

### **6. DATHybridRetriever**

Retriever híbrido com DAT (Dynamic Alpha Tuning) usando LLM judge.

```python
from src.retrievers.dat_hybrid import DATHybridRetriever
from src.retrievers.bm25_basic import BM25Basic
from src.retrievers.dense_faiss import DenseFaiss
from src.fusion.llm_judge import LLMJudge

bm25 = BM25Basic(k1=0.9, b=0.4)
dense = DenseFaiss(model_name="all-MiniLM-L6-v2", ...)
llm_judge = LLMJudge(model="gpt-4o-mini", ...)

retriever = DATHybridRetriever(
    bm25_retriever=bm25,
    dense_retriever=dense,
    llm_judge=llm_judge,
    weight_policy=DATWeightPolicy(),
    top_k=20,
)

retriever.build_index(documents)
results = retriever.retrieve(queries, k=10)
```

**Uso**: Quando você quer alpha adaptativo baseado em LLM judge.

---

### **7. BaselineHybridRetriever**

Retriever híbrido com alpha fixo (baseline).

```python
from src.retrievers.baseline_hybrid import BaselineHybridRetriever

retriever = BaselineHybridRetriever(
    bm25_retriever=bm25,
    dense_retriever=dense,
    alpha=0.6,  # Fixo: 60% dense, 40% BM25
    top_k=20,
)

retriever.build_index(documents)
results = retriever.retrieve(queries, k=10)
```

**Uso**: Baseline para comparação com DAT (alpha fixo vs adaptativo).

---

## ➕ **Como Adicionar um Novo Retriever**

### **Passo 1: Criar Classe que Implementa AbstractRetriever**

```python
# src/retrievers/my_retriever.py

from typing import Dict, List, Tuple
from ..core.interfaces import AbstractRetriever
from ..datasets.schema import Document, Query

class MyRetriever(AbstractRetriever):
    """Retriever customizado que faz X"""
    
    def __init__(self, param1: str = "default", param2: int = 100):
        self.param1 = param1
        self.param2 = param2
        self._index = {}  # Estrutura de índice
        self._doc_map = {}  # doc_id -> texto
    
    def build_index(self, docs: List[Document]) -> None:
        """Constrói índice a partir de documentos"""
        for doc in docs:
            doc_text = f"{doc.title} {doc.text}"
            # Sua lógica de indexação
            self._index[doc.doc_id] = self._process_doc(doc_text)
            self._doc_map[doc.doc_id] = doc_text
    
    def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Retorna top-K documentos para cada query"""
        results = {}
        for query in queries:
            # Sua lógica de busca
            scores = self._search(query.text)
            # Ordenar e retornar top-K
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            results[query.query_id] = sorted_scores[:k]
        return results
    
    def _process_doc(self, text: str):
        # Sua lógica de processamento
        pass
    
    def _search(self, query_text: str) -> Dict[str, float]:
        # Sua lógica de busca
        return {}
```

### **Passo 2: Registrar na Factory**

```python
# src/retrievers/factory.py

from .my_retriever import MyRetriever

def create_retriever(config: Dict[str, Any]) -> AbstractRetriever:
    retriever_type = config.get("type", "hybrid").lower()
    
    # ... código existente ...
    
    elif retriever_type == "my_retriever":
        return MyRetriever(
            param1=config.get("param1", "default"),
            param2=config.get("param2", 100),
        )
    
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
```

### **Passo 3: Atualizar Schema**

```python
# src/config/schema.py

class RetrieverConfig(BaseModel):
    type: Literal["hybrid", "dense", "tfidf", "graph", "bm25", "dat_hybrid", "baseline_hybrid", "my_retriever"]
    # ... outros campos ...
```

---

## ✅ **Boas Práticas**

### **1. Sempre Retorne Formato Consistente**

```python
# ✅ BOM - Formato correto
def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    return {
        "query_1": [("doc_1", 0.95), ("doc_2", 0.87)],
        "query_2": [("doc_3", 0.92), ("doc_1", 0.85)],
    }

# ❌ RUIM - Formato incorreto
def retrieve(self, queries: List[Query], k: int = 10) -> List[Tuple[str, float]]:
    return [("doc_1", 0.95), ("doc_2", 0.87)]  # Deveria ser Dict!
```

### **2. Ordene Resultados por Score (Descendente)**

```python
# ✅ BOM - Ordenado corretamente
results[query.query_id] = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

# ❌ RUIM - Não ordenado ou ordem errada
results[query.query_id] = list(scores.items())[:k]  # Pode não estar ordenado!
```

### **3. Cache Textos para Reranking (se aplicável)**

```python
# ✅ BOM - Cache textos para reranking
def build_index(self, docs: List[Document]) -> None:
    self._doc_map = {doc.doc_id: f"{doc.title} {doc.text}" for doc in docs}
    # ... construir índice ...

def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    candidates = self.index.search(query_vec, topk=150)
    # Reranking precisa dos textos
    cand_texts = [(doc_id, self._doc_map[doc_id]) for doc_id, _ in candidates]
    reranked = self.reranker.rescore(query.text, cand_texts, weights)
```

### **4. Trate Queries Vazias ou Inválidas**

```python
# ✅ BOM - Trata edge cases
def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    results = {}
    for query in queries:
        if not query.text or query.text.strip() == "":
            results[query.query_id] = []  # Retorna vazio para query vazia
            continue
        # ... busca normal ...

# ❌ RUIM - Pode quebrar com query vazia
def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    for query in queries:
        vec = self.vectorizer.encode_text(query.text)  # Pode dar erro se vazio!
```

### **5. Use Logging para Debug**

```python
# ✅ BOM - Logging útil
from ..utils.logging import get_logger

_log = get_logger("retriever.my_retriever")

def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    _log.info(f"Retrieving {len(queries)} queries with k={k}")
    # ...
    _log.debug(f"Query '{query.query_id}': {len(candidates)} candidates, top score={scores[0][1]:.4f}")
```

### **6. Valide k e Retorne Apropriadamente**

```python
# ✅ BOM - Valida k
def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    # ...
    results[query.query_id] = sorted_scores[:k]  # Garante que retorna no máximo k

# ❌ RUIM - Não valida k
def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    results[query.query_id] = sorted_scores  # Pode retornar mais que k!
```

---

## 🔍 **Pipeline de Retrieval (HybridRetriever)**

Fluxo detalhado do `HybridRetriever.retrieve()`:

1. **Encode Query**: `vectorizer.encode_text(query.text, is_query=True)` → fatias
2. **Concat**: `vectorizer.concat(fatias)` → vetor único
3. **First-Stage Search**: `index.search(query_vec, topk=150)` → candidatos
4. **Weight Policy**: `policy.weights(query.text)` → pesos adaptativos
5. **Reranking**: `reranker.rescore(query.text, candidates, weights)` → refinado
6. **Top-K Final**: `reranked[:k]` → resultados finais

**Por quê 2 estágios?**
- **First-stage (150)**: Busca rápida no índice completo
- **Reranking (150→10)**: Refinamento caro mas preciso nos candidatos

---

## 📚 **Referências**

- Veja `src/vectorizers/README.md` para vectorizers
- Veja `src/indexes/README.md` para indexes
- Veja `src/fusion/README.md` para fusion strategies