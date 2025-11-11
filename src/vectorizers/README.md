# 🔄 Vectorizers

Este módulo contém **vectorizers** que combinam múltiplos encoders para criar representações vetoriais multi-modais. Vectorizers são o ponto principal de extensão para criar novas soluções de retrieval.

---

## 📋 **Visão Geral**

Vectorizers são componentes que:

1. **Combinam encoders**: Usam múltiplos encoders (semantic, TF-IDF, entities) para criar representações ricas
2. **Fazem fit no corpus**: Coordenam o treinamento de encoders que precisam (TF-IDF, entities)
3. **Normalizam e escalam**: Aplicam normalizações e escalas para balancear diferentes modalidades
4. **Concatenam fatias**: Combinam vetores de diferentes modalidades em um único vetor

**Fluxo típico**:
```
Corpus → fit_corpus() → Treina encoders
Query → encode_text() → {"s": vec_s, "t": vec_t, "g": vec_g}
Fatias → concat() → [s; t; g] (normalizado L2)
```

---

## 🎯 **Vectorizers Disponíveis**

### **1. DenseVectorizer**

Apenas embeddings semânticos (sem TF-IDF ou entities).

```python
from src.vectorizers.dense_vectorizer import DenseVectorizer

vectorizer = DenseVectorizer(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
    query_prefix="",
    doc_prefix="",
    provider="huggingface",  # ou "openai"
    api_key=None,  # Para OpenAI
)

vectorizer.fit_corpus(["Doc 1...", "Doc 2..."])  # Apenas inicializa modelo
vec = vectorizer.concat(vectorizer.encode_text("query", is_query=True))
# Retorna: np.ndarray [384d]
```

**Uso**: Quando você quer apenas similaridade semântica, sem lexical ou entidades.

---

### **2. TFIDFVectorizer**

Apenas representação lexical TF-IDF.

```python
from src.vectorizers.tfidf_vectorizer import TFIDFVectorizer

vectorizer = TFIDFVectorizer(
    dim=1000,
    min_df=2,
    backend="sklearn",
)

vectorizer.fit_corpus(["Doc 1...", "Doc 2..."])  # Treina TF-IDF
vec = vectorizer.concat(vectorizer.encode_text("query", is_query=True))
# Retorna: np.ndarray [1000d]
```

**Uso**: Quando você quer apenas matching lexical (exato), sem semântica.

---

### **3. BiModalVectorizer**

Combina semantic + TF-IDF (2 modalidades).

```python
from src.vectorizers.bi_modal_vectorizer import BiModalVectorizer

vectorizer = BiModalVectorizer(
    semantic_model_name="sentence-transformers/all-MiniLM-L6-v2",
    tfidf_dim=1000,
    min_df=2,
    tfidf_backend="sklearn",
    device="cpu",
)

vectorizer.fit_corpus(["Doc 1...", "Doc 2..."])  # Treina TF-IDF
parts = vectorizer.encode_text("query", is_query=True)
# Retorna: {"s": [384d], "t": [1000d]}
vec = vectorizer.concat(parts)
# Retorna: np.ndarray [1384d]
```

**Uso**: Quando você quer combinar semântica + matching lexical.

---

### **4. TriModalVectorizer**

Combina semantic + TF-IDF + entities (3 modalidades).

```python
from src.vectorizers.tri_modal_vectorizer import TriModalVectorizer

vectorizer = TriModalVectorizer(
    semantic_model_name="sentence-transformers/all-MiniLM-L6-v2",
    tfidf_dim=1000,
    min_df=2,
    graph_model_name="BAAI/bge-large-en-v1.5",
    ner_backend="scispacy",
    device="cpu",
)

vectorizer.fit_corpus(["Doc 1...", "Doc 2..."])  # Treina TF-IDF + Entities
parts = vectorizer.encode_text("query", is_query=True)
# Retorna: {"s": [384d], "t": [1000d], "g": [1024d]}
vec = vectorizer.concat(parts)
# Retorna: np.ndarray [2408d]
```

**Uso**: Quando você quer máxima riqueza de representação (semântica + lexical + entidades).

---

### **5. GraphVectorizer**

Apenas embeddings de entidades.

```python
from src.vectorizers.graph_vectorizer import GraphVectorizer

vectorizer = GraphVectorizer(
    graph_model_name="BAAI/bge-large-en-v1.5",
    ner_backend="scispacy",
    device="cpu",
)

vectorizer.fit_corpus(["Doc 1...", "Doc 2..."])  # Treina Entities
vec = vectorizer.concat(vectorizer.encode_text("query", is_query=True))
# Retorna: np.ndarray [1024d]
```

**Uso**: Quando você quer apenas representação baseada em entidades.

---

## ➕ **Como Adicionar um Novo Vectorizer**

### **Passo 1: Criar Classe que Implementa AbstractVectorizer**

```python
# src/vectorizers/my_custom_vectorizer.py

from typing import Dict, Iterable
import numpy as np
from ..core.interfaces import AbstractVectorizer
from ..encoders.encoders import HFSemanticEncoder, TfidfEncoder, l2norm

class MyCustomVectorizer(AbstractVectorizer):
    """Vectorizer que combina BERT + TF-IDF customizado"""
    
    def __init__(
        self,
        semantic_model: str = "bert-base-uncased",
        tfidf_dim: int = 1000,
        min_df: int = 2,
    ):
        # Usar encoders existentes
        self.semantic = HFSemanticEncoder(model_name=semantic_model)
        self.tfidf = TfidfEncoder(dim=tfidf_dim, min_df=min_df, backend="sklearn")
        self.slice_dims = {}
        self.fitted = False
    
    def fit_corpus(self, docs_texts: Iterable[str]) -> None:
        """Treina encoders que precisam de fit"""
        docs_list = list(docs_texts)
        
        # TF-IDF precisa fit
        self.tfidf.fit(docs_list)
        
        # Semantic não precisa fit (stateless)
        self.slice_dims = {
            "s": int(self.semantic.dim or 768),
            "t": self.tfidf.vocab_size,
        }
        self.fitted = True
    
    def encode_text(self, text: str, is_query: bool = False) -> Dict[str, np.ndarray]:
        """Retorna fatias separadas"""
        assert self.fitted, "Chame fit_corpus() primeiro"
        
        s = self.semantic.encode_text(text, is_query=is_query)
        t = self.tfidf.encode_text(text)
        
        return {"s": s, "t": t}
    
    def concat(self, parts: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatena e normaliza"""
        vec = np.concatenate([parts["s"], parts["t"]]).astype(np.float32)
        return l2norm(vec)
    
    @property
    def total_dim(self) -> int:
        """Dimensão total após concatenação"""
        return self.slice_dims.get("s", 0) + self.slice_dims.get("t", 0)
```

### **Passo 2: Registrar na Factory**

```python
# src/vectorizers/factory.py

from .my_custom_vectorizer import MyCustomVectorizer

def create_vectorizer(config: Dict[str, Any]) -> AbstractVectorizer:
    vec_type = config.get("type", "tri_modal").lower()
    
    # ... código existente ...
    
    elif vec_type == "my_custom":
        return MyCustomVectorizer(
            semantic_model=config.get("semantic", {}).get("model", "bert-base-uncased"),
            tfidf_dim=config.get("tfidf", {}).get("dim", 1000),
            min_df=config.get("tfidf", {}).get("min_df", 2),
        )
    
    else:
        raise ValueError(f"Unknown vectorizer type: {vec_type}")
```

### **Passo 3: Atualizar Schema**

```python
# src/config/schema.py

class VectorizerConfig(BaseModel):
    type: Literal["dense", "tfidf", "bi_modal", "tri_modal", "graph", "my_custom"] = "tri_modal"
    # ... outros campos ...
```

### **Passo 4: Usar em YAML**

```yaml
retrievers:
  - type: "hybrid"
    vectorizer:
      type: "my_custom"
      semantic:
        model: "bert-base-uncased"
      tfidf:
        dim: 1000
        min_df: 2
```

---

## ✅ **Boas Práticas**

### **1. Sempre Chame fit_corpus() Antes de encode_text()**

```python
# ✅ BOM - Fit antes de encode
vectorizer.fit_corpus(docs)
vec = vectorizer.encode_text("query", is_query=True)

# ❌ RUIM - Encode sem fit
vec = vectorizer.encode_text("query", is_query=True)  # Pode dar erro!
```

### **2. Use Encoders Existentes Quando Possível**

```python
# ✅ BOM - Reutiliza encoders existentes
from ..encoders.encoders import HFSemanticEncoder, TfidfEncoder

class MyVectorizer(AbstractVectorizer):
    def __init__(self):
        self.semantic = HFSemanticEncoder(...)  # Reutiliza
        self.tfidf = TfidfEncoder(...)  # Reutiliza

# ❌ RUIM - Recria funcionalidade que já existe
class MyVectorizer(AbstractVectorizer):
    def __init__(self):
        # Recria TF-IDF do zero (desnecessário!)
        self._vocab = {}
        # ...
```

### **3. Normalize Vetores com l2norm()**

```python
# ✅ BOM - Usa função helper
from ..encoders.encoders import l2norm

def concat(self, parts: Dict[str, np.ndarray]) -> np.ndarray:
    vec = np.concatenate([parts["s"], parts["t"]])
    return l2norm(vec)

# ❌ RUIM - Normalização manual (pode ter bugs)
def concat(self, parts: Dict[str, np.ndarray]) -> np.ndarray:
    vec = np.concatenate([parts["s"], parts["t"]])
    return vec / np.linalg.norm(vec)  # Pode dar erro se norm=0
```

### **4. Documente Dimensões e Fatias**

```python
class MyVectorizer(AbstractVectorizer):
    """
    Vectorizer que combina X e Y.
    
    Fatias retornadas por encode_text():
    - "s": Semantic embedding [384d]
    - "t": TF-IDF [1000d]
    
    Dimensão total: 1384d
    """
    # ...
```

### **5. Trate Edge Cases (corpus vazio, texto vazio)**

```python
# ✅ BOM - Trata edge cases
def fit_corpus(self, docs_texts: Iterable[str]) -> None:
    docs_list = list(docs_texts)
    if not docs_list:
        raise ValueError("Corpus não pode estar vazio")
    self.tfidf.fit(docs_list)

# ❌ RUIM - Pode quebrar com corpus vazio
def fit_corpus(self, docs_texts: Iterable[str]) -> None:
    self.tfidf.fit(docs_texts)  # Pode dar erro se vazio
```

### **6. Use slice_dims para Rastrear Dimensões**

```python
# ✅ BOM - Rastreia dimensões claramente
def fit_corpus(self, docs_texts: Iterable[str]) -> None:
    self.tfidf.fit(docs_texts)
    self.slice_dims = {
        "s": int(self.semantic.dim or 384),
        "t": self.tfidf.vocab_size,
    }

@property
def total_dim(self) -> int:
    return sum(self.slice_dims.values())

# ❌ RUIM - Dimensões hardcoded ou calculadas incorretamente
@property
def total_dim(self) -> int:
    return 384 + 1000  # Hardcoded, pode estar errado!
```

---

## 📚 **Referências**

- Veja `src/encoders/README.md` para entender encoders disponíveis
- Veja `src/core/interfaces.py` para interface `AbstractVectorizer`