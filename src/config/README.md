# ⚙️ Configuration

Este módulo gerencia **carregamento, validação e merge de configurações** do framework. Todas as configurações são validadas usando Pydantic para garantir type safety e valores consistentes.

---

## 📋 **Visão Geral**

O módulo `config/` fornece:

- ✅ **Schemas Pydantic**: Validação automática de tipos e valores
- ✅ **Loaders**: Carregamento de YAML/JSON com merge de defaults
- ✅ **Type Safety**: Configurações tipadas e validadas em tempo de execução
- ✅ **CLI Overrides**: Suporte para sobrescrever valores via linha de comando

---

## 📁 **Estrutura**

```
config/
├── schema.py      # Modelos Pydantic (ExperimentConfig, RetrieverConfig, etc.)
├── loader.py      # Funções de carregamento (load_config, load_yaml, etc.)
├── defaults.py    # Valores padrão para configurações
└── __init__.py    # Exports públicos
```

---

## 🎯 **Componentes Principais**

### **1. Schemas (schema.py)**

Modelos Pydantic que definem a estrutura e validação de configurações.

#### **ExperimentConfig**

Configuração completa de um experimento.

```python
class ExperimentConfig(BaseModel):
    experiment: Dict[str, Any] = Field(default_factory=dict)
    dataset: Optional[DatasetConfig] = None
    datasets: Optional[List[DatasetConfig]] = None
    retrievers: List[RetrieverConfig]
    metrics: List[str] = Field(default_factory=lambda: ["nDCG", "MRR", "MAP", "Recall", "Precision"])
    ks: List[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    output_formats: List[str] = Field(default_factory=lambda: ["csv"])
    output_dir: Optional[str] = None
```

**Validações**:
- Pelo menos um retriever deve ser especificado
- Métricas devem estar na lista válida
- Dataset ou datasets deve ser especificado

#### **RetrieverConfig**

Configuração de um retriever individual.

```python
class RetrieverConfig(BaseModel):
    name: Optional[str] = None
    type: Literal["hybrid", "dense", "tfidf", "graph", "bm25", "dat_hybrid", "baseline_hybrid"]
    vectorizer: Optional[VectorizerConfig] = None
    fusion: Optional[FusionConfig] = None
    reranker: Optional[RerankerConfig] = None
    index: Optional[IndexConfig] = None
    # ... campos específicos por tipo
```

#### **VectorizerConfig**

Configuração de vectorizer.

```python
class VectorizerConfig(BaseModel):
    type: Literal["dense", "tfidf", "bi_modal", "tri_modal", "graph"]
    semantic: Optional[SemanticConfig] = None
    tfidf: Optional[TFIDFConfig] = None
    graph: Optional[GraphConfig] = None
```

---

### **2. Loader (loader.py)**

Funções para carregar e processar configurações.

#### **load_config()**

Carrega configuração de arquivo YAML/JSON e retorna `ExperimentConfig` validado.

```python
from src.config.loader import load_config

# Carregar de YAML
config = load_config("configs/my_experiment.yaml")

# Carregar de JSON
config = load_config("configs/my_experiment.json")

# Usar defaults
config = load_config()  # Retorna configuração padrão
```

**Comportamento**:
1. Carrega arquivo YAML/JSON
2. Faz merge com defaults (`defaults.py`)
3. Valida usando Pydantic
4. Retorna `ExperimentConfig` tipado

#### **load_yaml() / load_json()**

Carregam arquivos brutos sem validação.

```python
from src.config.loader import load_yaml, load_json

yaml_dict = load_yaml("config.yaml")
json_dict = load_json("config.json")
```

#### **merge_configs()**

Faz merge de dois dicionários de configuração.

```python
from src.config.loader import merge_configs

base = {"retrievers": [{"type": "dense"}]}
override = {"retrievers": [{"type": "hybrid"}]}
merged = merge_configs(base, override)
```

---

### **3. Defaults (defaults.py)**

Valores padrão para todas as configurações.

```python
from src.config.defaults import get_default_config

defaults = get_default_config()
# Retorna dict com valores padrão para todos os campos
```

---

## 📝 **Exemplo de Uso**

### **Carregar Configuração**

```python
from src.config.loader import load_config

# Carregar e validar
config = load_config("configs/scifact_experiment.yaml")

# Acessar campos tipados
print(config.experiment.get("name"))
print(config.retrievers[0].type)
print(config.metrics)
```

### **Criar Configuração Programaticamente**

```python
from src.config.schema import ExperimentConfig, RetrieverConfig, DatasetConfig

config = ExperimentConfig(
    experiment={"name": "my_experiment"},
    dataset=DatasetConfig(name="scifact"),
    retrievers=[
        RetrieverConfig(
            type="hybrid",
            vectorizer=VectorizerConfig(type="tri_modal"),
        )
    ],
    metrics=["nDCG", "MRR"],
    ks=[1, 5, 10],
)
```

### **Validar Configuração**

```python
from src.config.schema import ExperimentConfig

try:
    config = ExperimentConfig(**config_dict)
    print("✅ Configuração válida!")
except ValidationError as e:
    print(f"❌ Erro de validação: {e}")
```

---

## ✅ **Boas Práticas**

### **1. Sempre Use Schemas para Validação**

```python
# ✅ BOM - Validação automática
config = ExperimentConfig(**yaml_dict)

# ❌ RUIM - Sem validação
config = yaml_dict  # Pode ter erros silenciosos
```

### **2. Use Type Hints ao Trabalhar com Config**

```python
# ✅ BOM - Type hints claros
def process_config(config: ExperimentConfig) -> None:
    for retriever_config in config.retrievers:
        print(retriever_config.type)

# ❌ RUIM - Sem type hints
def process_config(config):
    for retriever_config in config.retrievers:
        print(retriever_config.type)
```

### **3. Valide Campos Opcionais Antes de Usar**

```python
# ✅ BOM - Verifica se existe antes de usar
if config.dataset:
    dataset_name = config.dataset.name
else:
    datasets = config.get_datasets()  # Usa método helper

# ❌ RUIM - Pode dar AttributeError
dataset_name = config.dataset.name  # Se dataset=None, quebra!
```

### **4. Use Literal Types para Valores Fixos**

```python
# ✅ BOM - Literal type garante valores válidos
type: Literal["hybrid", "dense", "tfidf"]

# ❌ RUIM - String genérica permite valores inválidos
type: str  # Pode receber "hybrid", "dense", ou "invalid"!
```

### **5. Documente Campos Complexos**

```python
class FusionConfig(BaseModel):
    """Configuração para estratégia de fusão.
    
    Args:
        strategy: Método de fusão ("weighted_cosine", "reciprocal_rank", etc.)
        policy: Política de pesos ("static", "heuristic", "dat")
        weights: Pesos fixos (apenas para policy="static")
        top_k: Top-K para retrieval antes de fusão (apenas para DAT)
    """
    strategy: Literal["weighted_cosine", "reciprocal_rank", "dat_linear"]
    policy: Literal["static", "heuristic", "dat"]
    weights: Optional[List[float]] = None
    top_k: Optional[int] = 20
```

---

## 🔧 **Adicionar Novo Campo de Configuração**

### **Passo 1: Atualizar Schema**

```python
# src/config/schema.py

class RetrieverConfig(BaseModel):
    # ... campos existentes ...
    my_new_field: Optional[str] = None  # Novo campo
```

### **Passo 2: Atualizar Defaults (se necessário)**

```python
# src/config/defaults.py

def get_default_config() -> Dict[str, Any]:
    return {
        # ... defaults existentes ...
        "retrievers": [{
            "my_new_field": "default_value",  # Default para novo campo
        }]
    }
```

### **Passo 3: Usar na Factory**

```python
# src/retrievers/factory.py

def create_retriever(config: Dict[str, Any]) -> AbstractRetriever:
    my_new_value = config.get("my_new_field", "default")
    # ...
```