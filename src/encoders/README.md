# 🔤 Encoders

Este módulo contém **encoders de baixo nível** que transformam texto em vetores para uma modalidade específica. Encoders são blocos reutilizáveis que podem ser compostos por vectorizers.

---

## 📋 **Visão Geral**

Encoders são componentes **stateless** (ou com estado mínimo) que fazem transformação direta:

```
Texto → Encoder → Vetor Numérico
```

**Características**:
- ✅ **Focados**: Cada encoder faz uma coisa (semantic, TF-IDF, entities)
- ✅ **Reutilizáveis**: Podem ser usados por múltiplos vectorizers
- ✅ **Baixo nível**: Não fazem fit de corpus complexo (exceto TF-IDF/Entities que precisam)
- ✅ **Composáveis**: Vectorizers combinam múltiplos encoders

---

## 🎯 **Encoders Disponíveis**

### **1. HFSemanticEncoder** (`encoders.py`)

Encoder semântico usando modelos Hugging Face (sentence-transformers ou transformers).

```python
from src.encoders.encoders import HFSemanticEncoder

encoder = HFSemanticEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",  # ou "cuda", "mps"
    query_prefix="",
    doc_prefix="",
)

# Encode (stateless, não precisa fit)
vec = encoder.encode_text("What is machine learning?", is_query=True)
# Retorna: np.ndarray [384d] (normalizado L2)
```

**Características**:
- Suporta sentence-transformers (preferencial) ou transformers + mean pooling
- Normalização L2 automática
- Prefixos diferentes para query/doc
- Dimensão detectada automaticamente

**Modelos suportados**: Qualquer modelo Hugging Face compatível com sentence-transformers ou transformers.

---

### **2. OpenAISemanticEncoder** (`encoders.py`)

Encoder semântico usando API da OpenAI (text-embedding-3-large, etc.).

```python
from src.encoders.encoders import OpenAISemanticEncoder

encoder = OpenAISemanticEncoder(
    model_name="text-embedding-3-large",
    api_key="sk-...",  # ou via OPENAI_API_KEY env var
)

vec = encoder.encode_text("What is machine learning?", is_query=True)
# Retorna: np.ndarray [3072d] (normalizado L2)
```

**Características**:
- Requer `OPENAI_API_KEY` (env var ou parâmetro)
- Suporte a batch encoding (`encode_batch()`)
- Dimensões: 3072 (large), 1536 (small/ada-002)

---

### **3. TfidfEncoder** (`encoders.py`)

Encoder TF-IDF com backends sklearn ou pyserini.

```python
from src.encoders.encoders import TfidfEncoder

encoder = TfidfEncoder(
    dim=1000,
    min_df=2,
    backend="sklearn",  # ou "pyserini"
    language="english",
)

# Precisa fit no corpus
encoder.fit(["Documento 1...", "Documento 2..."])

# Encode
vec = encoder.encode_text("machine learning")
# Retorna: np.ndarray [1000d] (normalizado L2)
```

**Características**:
- **Stateful**: Precisa `fit()` antes de `encode_text()`
- Backend sklearn: tokenização básica
- Backend pyserini: tokenização Lucene (mais robusta)
- Normalização L2 automática

---

### **4. EntityEncoderReal** (`entity_encoder.py`)

Encoder de entidades usando NER (spaCy/scispaCy) + embeddings de entidades.

```python
from src.encoders.entity_encoder import EntityEncoderReal, NERConfig, CacheConfig

ner_cfg = NERConfig(
    backend="scispacy",  # ou "spacy", "none"
    model="en_ner_bc5cdr_md",
    use_noun_chunks=True,
    batch_size=128,
)

cache_cfg = CacheConfig(
    artifact_dir=Path("./artifacts/entities"),
    force_rebuild=False,
)

encoder = EntityEncoderReal(
    graph_model_name="BAAI/bge-large-en-v1.5",
    device="cpu",
    ner=ner_cfg,
    min_df=2,
    max_entities_per_text=128,
    cache=cache_cfg,
)

# Precisa fit no corpus
encoder.fit(["Documento 1...", "Documento 2..."])

# Encode
vec = encoder.encode_text("Machine learning is used in AI")
# Retorna: np.ndarray [1024d] (agregação TF-IDF de embeddings de entidades)
```

**Características**:
- **Stateful**: Precisa `fit()` para calcular IDF de entidades
- Extrai entidades via NER (spaCy/scispaCy)
- Embute cada entidade com modelo HF
- Agrega por TF-IDF: `g(text) = L2(sum_e [tf(e,text) * idf(e) * emb(e)])`
- Cache persistente de embeddings de entidades

---

## 🔧 **Como Usar Encoders em Vectorizers**

### **Exemplo: Vectorizer Customizado**

```python
from src.vectorizers.my_vectorizer import AbstractVectorizer
from src.encoders.encoders import HFSemanticEncoder, TfidfEncoder, l2norm
import numpy as np

class MyBiModalVectorizer(AbstractVectorizer):
    def __init__(self, semantic_model: str = "all-MiniLM-L6-v2"):
        # Usar encoders existentes
        self.semantic = HFSemanticEncoder(model_name=semantic_model)
        self.tfidf = TfidfEncoder(dim=1000, min_df=2)
        self.slice_dims = {}
        self.fitted = False
    
    def fit_corpus(self, docs_texts: Iterable[str]) -> None:
        docs_list = list(docs_texts)
        # Fit apenas TF-IDF (semantic não precisa)
        self.tfidf.fit(docs_list)
        self.slice_dims = {
            "s": int(self.semantic.dim or 384),
            "t": self.tfidf.vocab_size,
        }
        self.fitted = True
    
    def encode_text(self, text: str, is_query: bool = False) -> Dict[str, np.ndarray]:
        assert self.fitted, "Chame fit_corpus() primeiro"
        s = self.semantic.encode_text(text, is_query=is_query)
        t = self.tfidf.encode_text(text)
        return {"s": s, "t": t}
    
    def concat(self, parts: Dict[str, np.ndarray]) -> np.ndarray:
        vec = np.concatenate([parts["s"], parts["t"]]).astype(np.float32)
        return l2norm(vec)
    
    @property
    def total_dim(self) -> int:
        return self.slice_dims["s"] + self.slice_dims["t"]
```

---

## ➕ **Como Adicionar um Novo Encoder**

### **Passo 1: Criar Classe do Encoder**

```python
# src/encoders/my_encoder.py

import numpy as np
from typing import Optional

class MyCustomEncoder:
    """Encoder customizado que faz X"""
    
    def __init__(self, param1: str = "default", param2: int = 100):
        self.param1 = param1
        self.param2 = param2
        self.dim = 512  # Dimensão do vetor de saída
        self._fitted = False  # Se precisar fit
    
    def fit(self, documents: Iterable[str]) -> None:
        """Treina encoder no corpus (se necessário)"""
        # Sua lógica de fit
        self._fitted = True
    
    def encode_text(self, text: str, is_query: bool = False) -> np.ndarray:
        """Converte texto em vetor"""
        # Sua lógica de encoding
        vec = np.zeros(self.dim, dtype=np.float32)
        # ... preencher vec ...
        return vec / (np.linalg.norm(vec) + 1e-8)  # Normalizar L2
```

### **Passo 2: Usar no Vectorizer**

```python
from src.encoders.my_encoder import MyCustomEncoder

class MyVectorizer(AbstractVectorizer):
    def __init__(self):
        self.my_encoder = MyCustomEncoder(param1="value")
        # ...
```

---

## ✅ **Boas Práticas**

### **1. Encoders Devem Ser Stateless (quando possível)**

```python
# ✅ BOM - Stateless (não precisa fit)
class SemanticEncoder:
    def encode_text(self, text: str) -> np.ndarray:
        return self.model.encode(text)

# ⚠️ OK - Stateful quando necessário (TF-IDF, Entities)
class TfidfEncoder:
    def fit(self, docs: Iterable[str]) -> None:
        # Precisa fit para vocabulário
        pass
```

### **2. Sempre Normalize Vetores (L2)**

```python
# ✅ BOM - Normalização L2
def encode_text(self, text: str) -> np.ndarray:
    vec = self._compute_vector(text)
    return vec / (np.linalg.norm(vec) + 1e-8)

# ❌ RUIM - Sem normalização
def encode_text(self, text: str) -> np.ndarray:
    return self._compute_vector(text)  # Pode ter magnitudes diferentes
```

### **3. Documente Dimensões e Requisitos**

```python
class MyEncoder:
    """
    Encoder que faz X.
    
    Dimensão de saída: 512
    Requer fit: Não (stateless)
    Normalização: L2 automática
    """
    def __init__(self):
        self.dim = 512
```

### **4. Trate Casos Edge (texto vazio, None)**

```python
# ✅ BOM - Trata edge cases
def encode_text(self, text: str, is_query: bool = False) -> np.ndarray:
    if not text or text.strip() == "":
        return np.zeros(self.dim, dtype=np.float32)
    # ... encoding normal ...

# ❌ RUIM - Pode quebrar com texto vazio
def encode_text(self, text: str, is_query: bool = False) -> np.ndarray:
    return self.model.encode(text)  # Pode dar erro se text=""
```

### **5. Use Type Hints Consistentes**

```python
# ✅ BOM - Type hints claros
def encode_text(self, text: str, is_query: bool = False) -> np.ndarray:
    pass

# ❌ RUIM - Sem type hints
def encode_text(self, text, is_query=False):
    pass
```