# 🔍 Indexes

Este módulo contém **estruturas de busca vetorial** que armazenam e buscam vetores de documentos de forma eficiente. Indexes são responsáveis pela fase de busca rápida (first-stage retrieval).

---

## 📋 **Visão Geral**

Indexes recebem vetores já criados pelo `Vectorizer` e os organizam em estruturas otimizadas para busca por similaridade:

```
Vetores de Documentos → Index.build() → Estrutura de Busca
Query Vector → Index.search() → Top-K Documentos
```

**Características**:
- ✅ **Não criam vetores**: Recebem vetores prontos do vectorizer
- ✅ **Busca eficiente**: Otimizados para top-K retrieval rápido
- ✅ **Persistência**: Podem salvar/carregar índices em disco
- ✅ **GPU Support**: Alguns suportam aceleração GPU (FAISS)

---

## 🎯 **Indexes Disponíveis**

### **1. FaissFlatIPIndex** (`faiss_index.py`)

Índice FAISS simples com busca exata (IndexFlatIP - Inner Product).

```python
from src.indexes.faiss_index import FaissFlatIPIndex

index = FaissFlatIPIndex(
    artifact_dir="./artifacts",
    index_name="dense.index",
)

# Construir índice
doc_ids = ["doc1", "doc2", ...]
doc_vectors = np.array([...])  # Shape: (n_docs, dim)
index.build_from_matrix(doc_ids, doc_vectors)

# Buscar
query_vec = np.array([...])  # Shape: (dim,)
results = index.search(query_vec, topk=10)
# Retorna: [("doc1", 0.95), ("doc2", 0.87), ...]
```

**Características**:
- Busca exata (não aproximada)
- Inner Product (IP) como métrica (≈ cosine para vetores L2-normalizados)
- Persistência automática (salva/carrega de disco)
- Fallback NumPy se FAISS não disponível

**Uso**: Quando você quer busca exata e precisa de resultados precisos.

---

### **2. HybridIndex** (`hybrid_index.py`)

Índice FAISS avançado com suporte a busca aproximada e GPU.

```python
from src.indexes.hybrid_index import HybridIndex
from src.vectorizers.tri_modal_vectorizer import TriModalVectorizer

vectorizer = TriModalVectorizer(...)
vectorizer.fit_corpus(docs_texts)

index = HybridIndex(
    vectorizer=vectorizer,
    faiss_factory=None,  # IndexFlatIP (exato)
    # faiss_factory="OPQ64,IVF4096,PQ64x8",  # Busca aproximada
    faiss_metric="ip",  # "ip" ou "l2"
    faiss_nprobe=64,  # Para IVF (número de clusters a buscar)
    faiss_train_size=0,  # 0 = auto
    artifact_dir="./artifacts",
    index_name="hybrid.index",
)

# Construir índice (vetoriza internamente)
index.build([(doc_id, doc_text) for doc_id, doc_text in docs])

# Buscar
query_vec = vectorizer.concat(vectorizer.encode_text("query", is_query=True))
results = index.search(query_vec, topk=150)
# Retorna: [("doc1", 0.95), ("doc2", 0.87), ...]
```

**Características**:
- **Busca exata**: `faiss_factory=None` → IndexFlatIP
- **Busca aproximada**: `faiss_factory="OPQ64,IVF4096,PQ64x8"` → Index comprimido
- **GPU acceleration**: Cria cópia GPU automaticamente se disponível
- **Persistência**: Salva/carrega índice e doc_ids
- **Auto-train**: Calcula tamanho de treino automaticamente para índices que precisam

**Factory Strings Comuns**:
- `None` ou `"FlatIP"`: Busca exata (mais lento, mais preciso)
- `"IVF4096,Flat"`: Inverted File Index (mais rápido, menos preciso)
- `"OPQ64,IVF4096,PQ64x8"`: Product Quantization (muito rápido, menos preciso)

**Uso**: Quando você quer flexibilidade entre precisão e velocidade.

---

## ➕ **Como Adicionar um Novo Index**

### **Passo 1: Criar Classe que Implementa AbstractIndex**

```python
# src/indexes/annoy_index.py

from typing import List, Tuple, Iterable, Optional
import numpy as np
from pathlib import Path
import json
from annoy import AnnoyIndex
from ..core.interfaces import AbstractIndex, AbstractVectorizer

class AnnoyIndexWrapper(AbstractIndex):
    """Wrapper para Annoy (biblioteca de busca aproximada)"""
    
    def __init__(
        self,
        vectorizer: AbstractVectorizer,
        n_trees: int = 10,
        metric: str = "angular",  # "angular" ou "euclidean"
        artifact_dir: Optional[str] = None,
    ):
        self.vec = vectorizer
        self.n_trees = n_trees
        self.metric = metric
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.index = None
        self.doc_ids: List[str] = []
        self.dimension = 0
    
    def build(self, doc_id_and_text: Iterable[Tuple[str, str]]) -> None:
        """Constrói índice Annoy"""
        doc_list = list(doc_id_and_text)
        
        # Vetorizar documentos
        vectors = []
        for doc_id, text in doc_list:
            parts = self.vec.encode_text(text, is_query=False)
            vec = self.vec.concat(parts)
            vectors.append(vec)
            self.doc_ids.append(doc_id)
        
        if not vectors:
            return
        
        # Criar índice Annoy
        self.dimension = len(vectors[0])
        metric_type = AnnoyIndex.ANGULAR if self.metric == "angular" else AnnoyIndex.EUCLIDEAN
        self.index = AnnoyIndex(self.dimension, metric_type)
        
        for i, vec in enumerate(vectors):
            self.index.add_item(i, vec)
        
        # Construir árvores
        self.index.build(self.n_trees)
        
        # Salvar se artifact_dir especificado
        if self.artifact_dir:
            self._save()
    
    def search(self, query_vec: np.ndarray, topk: int = 10) -> List[Tuple[str, float]]:
        """Busca top-K"""
        if self.index is None:
            return []
        
        # Annoy retorna (indices, distances)
        indices, distances = self.index.get_nns_by_vector(
            query_vec.tolist(),
            topk,
            include_distances=True
        )
        
        # Converter distâncias em scores
        # Para angular: score = 1 - distance
        scores = [1.0 - d for d in distances] if self.metric == "angular" else distances
        
        return [(self.doc_ids[i], float(score)) for i, score in zip(indices, scores)]
    
    def _save(self):
        """Salva índice em disco"""
        if self.artifact_dir and self.index:
            index_path = self.artifact_dir / "annoy.index"
            self.index.save(str(index_path))
            
            # Salvar doc_ids
            ids_path = self.artifact_dir / "annoy.ids.json"
            with open(ids_path, "w") as f:
                json.dump(self.doc_ids, f)
```

### **Passo 2: Registrar na Factory**

```python
# src/indexes/factory.py

from .annoy_index import AnnoyIndexWrapper

def create_index(config: Dict[str, Any], vectorizer: AbstractVectorizer) -> AbstractIndex:
    index_type = config.get("type", "faiss").lower()
    
    # ... código existente ...
    
    elif index_type == "annoy":
        return AnnoyIndexWrapper(
            vectorizer=vectorizer,
            n_trees=config.get("n_trees", 10),
            metric=config.get("metric", "angular"),
            artifact_dir=config.get("artifact_dir"),
        )
```

### **Passo 3: Atualizar Schema**

```python
# src/config/schema.py

class IndexConfig(BaseModel):
    type: Literal["faiss", "numpy", "annoy"] = "faiss"
    # ... outros campos ...
```

---

## ✅ **Boas Práticas**

### **1. Sempre Vetorize Documentos no build()**

```python
# ✅ BOM - Vetoriza internamente
def build(self, doc_id_and_text: Iterable[Tuple[str, str]]) -> None:
    for doc_id, text in doc_id_and_text:
        vec = self.vec.concat(self.vec.encode_text(text, is_query=False))
        # Adiciona ao índice

# ❌ RUIM - Espera vetores prontos (quebra abstração)
def build(self, doc_vectors: np.ndarray) -> None:
    # Não deveria receber vetores, deveria receber textos!
    pass
```

### **2. Use Persistência para Índices Grandes**

```python
# ✅ BOM - Salva índice para reuso
index = HybridIndex(
    artifact_dir="./artifacts",
    index_name="my_index",
)
index.build(docs)  # Salva automaticamente

# Carregar depois
if index.try_load():
    print("Índice carregado do cache!")

# ❌ RUIM - Reconstrói sempre (lento)
index.build(docs)  # Sem persistência, sempre reconstrói
```

### **3. Escolha Factory Baseado em Trade-off Precisão/Velocidade**

```python
# ✅ BOM - Escolhe factory apropriada
if n_docs < 100_000:
    factory = None  # FlatIP (exato, rápido para pequenos)
else:
    factory = "IVF4096,Flat"  # Aproximado (rápido para grandes)

# ❌ RUIM - Sempre usa FlatIP (pode ser muito lento)
factory = None  # Para 1M docs, pode ser lento demais
```

### **4. Trate Casos de Índice Vazio**

```python
# ✅ BOM - Trata índice vazio
def search(self, query_vec: np.ndarray, topk: int = 10) -> List[Tuple[str, float]]:
    if self.index is None or len(self.doc_ids) == 0:
        return []
    # ... busca normal ...

# ❌ RUIM - Pode quebrar com índice vazio
def search(self, query_vec: np.ndarray, topk: int = 10) -> List[Tuple[str, float]]:
    return self.index.search(query_vec, topk)  # Pode dar erro se vazio
```

### **5. Normalize Query Vector (se necessário)**

```python
# ✅ BOM - Garante que query está normalizado
def search(self, query_vec: np.ndarray, topk: int = 10) -> List[Tuple[str, float]]:
    # Se usar Inner Product, vetores devem estar L2-normalizados
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    return self.index.search(query_vec, topk)

# Nota: Na prática, vectorizers já normalizam, mas é bom garantir
```

---

## 🔍 **Métricas de Similaridade**

### **Inner Product (IP)**

- **Fórmula**: `score = dot(query_vec, doc_vec)`
- **Quando usar**: Com vetores L2-normalizados, IP ≈ cosine similarity
- **Vantagem**: Mais rápido que cosine (não precisa normalizar durante busca)

### **L2 Distance**

- **Fórmula**: `distance = ||query_vec - doc_vec||²`
- **Quando usar**: Quando você quer distância euclidiana
- **Nota**: Menor distância = maior similaridade (inverter para scores)

---

## 📚 **Referências**

- **FAISS**: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- Veja `src/vectorizers/README.md` para entender vectorizers