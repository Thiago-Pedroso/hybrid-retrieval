# 🔌 Core Interfaces

Este módulo define as **interfaces abstratas (ABCs)** que todos os componentes do framework devem implementar. Essas interfaces garantem que diferentes implementações sejam intercambiáveis e sigam contratos consistentes.

---

## 📋 **Visão Geral**

O módulo `core/` contém apenas **interfaces**, não implementações. É a base do sistema de design baseado em contratos que permite:

- ✅ **Modularidade**: Trocar implementações sem quebrar código
- ✅ **Testabilidade**: Mockar interfaces facilmente
- ✅ **Type Safety**: Type hints claros para IDEs e ferramentas
- ✅ **Documentação**: Contratos explícitos do que cada componente deve fazer

---

## 🎯 **Interfaces Disponíveis**

### **1. AbstractRetriever**

Interface para sistemas de retrieval que podem indexar documentos e buscar resultados.

```python
class AbstractRetriever(ABC):
    @abstractmethod
    def build_index(self, docs: List[Document]) -> None:
        """Constrói índice a partir de documentos"""
    
    @abstractmethod
    def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Retorna top-K documentos para cada query"""
```

**Contrato**:
- `build_index()`: Deve processar todos os documentos e preparar estrutura de busca
- `retrieve()`: Deve retornar dicionário `{query_id: [(doc_id, score), ...]}` ordenado por score (descendente)

**Implementações**: `HybridRetriever`, `DenseFaiss`, `BM25Basic`, `DATHybridRetriever`, etc.

---

### **2. AbstractVectorizer**

Interface para conversão de texto em vetores numéricos.

```python
class AbstractVectorizer(ABC):
    @abstractmethod
    def fit_corpus(self, docs_texts: Iterable[str]) -> None:
        """Treina vectorizer no corpus"""
    
    @abstractmethod
    def encode_text(self, text: str, is_query: bool = False) -> Dict[str, np.ndarray]:
        """Converte texto em vetor(es) - pode retornar múltiplas fatias"""
    
    @abstractmethod
    def concat(self, parts: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatena e normaliza fatias"""
    
    @property
    @abstractmethod
    def total_dim(self) -> int:
        """Dimensão total do vetor concatenado"""
```

**Contrato**:
- `fit_corpus()`: Deve ser chamado antes de `encode_text()` para treinar vocabulário/IDF/etc
- `encode_text()`: Pode retornar múltiplas fatias (ex: `{"s": vec_s, "t": vec_t, "g": vec_g}`)
- `concat()`: Deve concatenar fatias e normalizar (tipicamente L2)
- `total_dim`: Propriedade que retorna dimensão final após concatenação

**Implementações**: `TriModalVectorizer`, `DenseVectorizer`, `TFIDFVectorizer`, etc.

---

### **3. AbstractIndex**

Interface para estruturas de busca vetorial.

```python
class AbstractIndex(ABC):
    @abstractmethod
    def build(self, doc_id_and_text: Iterable[Tuple[str, str]]) -> None:
        """Constrói índice a partir de documentos"""
    
    @abstractmethod
    def search(self, query_vec: np.ndarray, topk: int = 10) -> List[Tuple[str, float]]:
        """Busca top-K vetores mais similares"""
```

**Contrato**:
- `build()`: Recebe pares `(doc_id, text)` e constrói estrutura de busca
- `search()`: Recebe vetor de query e retorna lista `[(doc_id, score), ...]` ordenada por score (descendente)

**Implementações**: `HybridIndex`, `FaissFlatIPIndex`

---

### **4. AbstractFusionStrategy**

Interface para combinar múltiplas listas ranqueadas.

```python
class AbstractFusionStrategy(ABC):
    @abstractmethod
    def fuse(
        self,
        query: str,
        results_list: List[Dict[str, List[Tuple[str, float]]]],
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """Combina múltiplas listas ranqueadas em uma única"""
```

**Contrato**:
- `fuse()`: Recebe lista de resultados (uma por modalidade/método) e retorna lista única combinada
- `weights`: Pesos opcionais para cada lista (se None, usa pesos iguais)

**Implementações**: `WeightedCosineFusion`, `ReciprocalRankFusion`, `DATLinearFusion`

---

### **5. AbstractWeightPolicy**

Interface para políticas de pesos adaptativos.

```python
class AbstractWeightPolicy(ABC):
    @abstractmethod
    def weights(self, query_text: str) -> Tuple[float, ...]:
        """Retorna pesos para cada modalidade baseado na query"""
```

**Contrato**:
- `weights()`: Retorna tupla de pesos (um por modalidade), tipicamente normalizados

**Implementações**: `StaticPolicy`, `HeuristicLLMPolicy`, `DATWeightPolicy`

---

### **6. AbstractReranker**

Interface para reranking de candidatos.

```python
class AbstractReranker(ABC):
    @abstractmethod
    def rescore(
        self,
        query_text: str,
        candidate_docs: List[Tuple[str, str]],
        weights: Optional[Tuple[float, ...]] = None,
    ) -> List[Tuple[str, float]]:
        """Recalcula scores dos candidatos"""
```

**Contrato**:
- `rescore()`: Recebe query e lista de candidatos `(doc_id, doc_text)` e retorna lista reordenada

**Implementações**: `TriModalReranker`, `BiModalReranker`

---

### **7. AbstractMetric**

Interface para métricas de avaliação.

```python
class AbstractMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Nome da métrica"""
    
    @abstractmethod
    def compute(
        self,
        ranked: List[str],
        gold: Dict[str, float],
        k: int = 10,
    ) -> float:
        """Calcula métrica para uma query"""
```

**Contrato**:
- `name`: Propriedade que retorna nome da métrica (ex: "nDCG", "MRR")
- `compute()`: Recebe lista ranqueada, dicionário de relevância `{doc_id: score}` e cutoff `k`

**Implementações**: `NDCGMetric`, `MRRMetric`, `MAPMetric`, `RecallMetric`, `PrecisionMetric`

---

### **8. AbstractOutputFormatter**

Interface para formatadores de saída.

```python
class AbstractOutputFormatter(ABC):
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Nome do formato"""
    
    @abstractmethod
    def format(
        self,
        results: Any,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Formata resultados"""
```

**Implementações**: `CSVFormatter`, `JSONFormatter`, `JSONLFormatter`

---

## ✅ **Boas Práticas**

### **1. Sempre Implemente Todas as Métodos Abstratos**

```python
# ✅ BOM
class MyRetriever(AbstractRetriever):
    def build_index(self, docs: List[Document]) -> None:
        # Implementação completa
        pass
    
    def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        # Implementação completa
        pass

# ❌ RUIM - Falta implementar método abstrato
class MyRetriever(AbstractRetriever):
    def build_index(self, docs: List[Document]) -> None:
        pass
    # Faltou retrieve()!
```

### **2. Respeite os Tipos de Retorno**

```python
# ✅ BOM - Retorna tipo exato da interface
def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    return {"q1": [("doc1", 0.95), ("doc2", 0.87)]}

# ❌ RUIM - Retorna tipo diferente
def retrieve(self, queries: List[Query], k: int = 10) -> List[Tuple[str, float]]:
    return [("doc1", 0.95), ("doc2", 0.87)]  # Deveria ser Dict!
```

### **3. Documente Comportamentos Específicos**

```python
# ✅ BOM - Documenta comportamento específico
class MyVectorizer(AbstractVectorizer):
    def encode_text(self, text: str, is_query: bool = False) -> Dict[str, np.ndarray]:
        """
        Retorna fatias separadas.
        
        Nota: Este vectorizer sempre retorna {"s": semantic_vec} mesmo que
        is_query=True, pois não diferencia query/doc.
        """
        return {"s": self.semantic.encode(text)}
```

### **4. Valide Entradas Quando Apropriado**

```python
# ✅ BOM - Valida estado antes de processar
def encode_text(self, text: str, is_query: bool = False) -> Dict[str, np.ndarray]:
    if not self.fitted:
        raise RuntimeError("Chame fit_corpus() antes de encode_text()")
    # ...
```

### **5. Use Type Hints Consistentes**

```python
# ✅ BOM - Type hints claros
def search(self, query_vec: np.ndarray, topk: int = 10) -> List[Tuple[str, float]]:
    pass

# ❌ RUIM - Type hints genéricos ou ausentes
def search(self, query_vec, topk=10):
    pass
```

---

## 🔍 **Como Usar Interfaces**

### **Para Implementar um Componente Novo**

1. Importe a interface correspondente
2. Herde da classe abstrata
3. Implemente todos os métodos abstratos
4. Adicione na factory correspondente

```python
from ..core.interfaces import AbstractRetriever
from ..datasets.schema import Document, Query

class MyRetriever(AbstractRetriever):
    def build_index(self, docs: List[Document]) -> None:
        # Sua implementação
        pass
    
    def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        # Sua implementação
        return {"q1": [("doc1", 0.95)]}
```

### **Para Testar Componentes**

Interfaces facilitam mocking em testes:

```python
from unittest.mock import Mock
from ..core.interfaces import AbstractRetriever

def test_my_pipeline():
    # Mock do retriever
    mock_retriever = Mock(spec=AbstractRetriever)
    mock_retriever.retrieve.return_value = {"q1": [("doc1", 0.95)]}
    
    # Usar mock no pipeline
    results = my_pipeline(mock_retriever)
    assert results["q1"][0][0] == "doc1"
```

---

## 📚 **Referências**

- **ABCs do Python**: [https://docs.python.org/3/library/abc.html](https://docs.python.org/3/library/abc.html)
- **Type Hints**: [https://docs.python.org/3/library/typing.html](https://docs.python.org/3/library/typing.html)