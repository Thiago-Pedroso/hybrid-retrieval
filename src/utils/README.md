# 🛠️ Utils

Este módulo contém **utilitários auxiliares** usados em todo o framework: logging, I/O, e outras funções helper.

---

## 📋 **Visão Geral**

O módulo `utils/` fornece:

- ✅ **Logging**: Sistema de logging estruturado com progresso e timing
- ✅ **I/O**: Funções para criar diretórios e gerenciar arquivos
- ✅ **Helpers**: Funções auxiliares reutilizáveis

---

## 🎯 **Componentes**

### **1. Logging** (`logging.py`)

Sistema de logging com suporte a progresso e timing.

#### **get_logger()**

Cria logger com nome específico.

```python
from src.utils.logging import get_logger

_log = get_logger("my_module")

_log.info("Mensagem informativa")
_log.debug("Mensagem de debug")
_log.warning("Aviso")
_log.error("Erro")
```

**Nomes de logger recomendados**:
- `"retriever.hybrid"` para retrievers
- `"vectorizer.tri_modal"` para vectorizers
- `"index.faiss"` para indexes
- `"experiment.runner"` para experiments

---

#### **log_time()**

Context manager para medir tempo de execução.

```python
from src.utils.logging import log_time, get_logger

_log = get_logger("my_module")

with log_time(_log, "Processar documentos"):
    # Código que você quer medir
    process_documents(docs)

# Output: "⏱️  Processar documentos: 2.34s"
```

**Uso**: Para medir performance de operações longas.

---

#### **ProgressLogger**

Logger de progresso com barra visual.

```python
from src.utils.logging import ProgressLogger, get_logger

_log = get_logger("my_module")

with ProgressLogger(_log, "Processando", total=1000, log_every=100) as progress:
    for i in range(1000):
        # Processar item
        process_item(i)
        progress.update(1)

# Output:
# "🔄 Processando: 100/1000 (10%)"
# "🔄 Processando: 200/1000 (20%)"
# ...
```

**Parâmetros**:
- `total`: Número total de itens
- `log_every`: Frequência de log (a cada N itens)

---

### **2. I/O** (`io.py`)

Funções para gerenciar arquivos e diretórios.

#### **ensure_dir()**

Garante que diretório existe (cria se não existir).

```python
from src.utils.io import ensure_dir
from pathlib import Path

# Criar diretório
ensure_dir(Path("./outputs/experiments"))

# Ou passar string
ensure_dir("./outputs/experiments")
```

**Uso**: Antes de salvar arquivos, garantir que diretório existe.

---

## ✅ **Boas Práticas**

### **1. Use Loggers Específicos por Módulo**

```python
# ✅ BOM - Logger específico
from src.utils.logging import get_logger

_log = get_logger("retriever.my_retriever")

def retrieve(self, queries, k):
    _log.info(f"Retrieving {len(queries)} queries")

# ❌ RUIM - Logger genérico
import logging
_log = logging.getLogger(__name__)  # Menos controle
```

### **2. Use log_time() para Operações Longas**

```python
# ✅ BOM - Mede tempo
with log_time(_log, "Build index"):
    self.index.build(docs)

# ❌ RUIM - Sem medição de tempo
self.index.build(docs)  # Não sabe quanto tempo levou
```

### **3. Use ProgressLogger para Loops Longos**

```python
# ✅ BOM - Mostra progresso
with ProgressLogger(_log, "Encoding", total=len(docs), log_every=100) as p:
    for doc in docs:
        encode(doc)
        p.update(1)

# ❌ RUIM - Sem feedback
for doc in docs:
    encode(doc)  # Usuário não sabe progresso
```

### **4. Sempre Use ensure_dir() Antes de Salvar**

```python
# ✅ BOM - Garante diretório existe
from src.utils.io import ensure_dir

output_path = Path("./outputs/results.csv")
ensure_dir(output_path.parent)
output_path.write_text(data)

# ❌ RUIM - Pode dar erro se diretório não existe
output_path = Path("./outputs/results.csv")
output_path.write_text(data)  # FileNotFoundError se ./outputs não existe!
```

### **5. Use Níveis de Log Apropriados**

```python
# ✅ BOM - Níveis apropriados
_log.info("Iniciando processamento")  # Info geral
_log.debug(f"Query vec shape: {vec.shape}")  # Debug detalhado
_log.warning("Cache não encontrado, reconstruindo")  # Aviso
_log.error("Erro ao carregar modelo", exc_info=True)  # Erro com traceback

# ❌ RUIM - Tudo como info
_log.info("Iniciando")
_log.info(f"Query vec shape: {vec.shape}")  # Debug deveria ser debug
_log.info("Erro!")  # Erro deveria ser error
```

---

## 🔍 **Exemplos de Uso**

### **Logging em Retriever**

```python
from src.utils.logging import get_logger, log_time, ProgressLogger

_log = get_logger("retriever.my_retriever")

class MyRetriever(AbstractRetriever):
    def build_index(self, docs: List[Document]) -> None:
        _log.info(f"🚀 Building index for {len(docs)} documents")
        
        with log_time(_log, "Encode documents"):
            vectors = []
            with ProgressLogger(_log, "Encoding", total=len(docs), log_every=100) as p:
                for doc in docs:
                    vec = self.vectorizer.encode(doc.text)
                    vectors.append(vec)
                    p.update(1)
        
        _log.info(f"✅ Index built successfully")
    
    def retrieve(self, queries: List[Query], k: int = 10):
        _log.info(f"🔍 Retrieving {len(queries)} queries with k={k}")
        # ...
        _log.debug(f"Query '{query.query_id}': top score={results[0][1]:.4f}")
```

### **I/O em Experiment**

```python
from src.utils.io import ensure_dir
from pathlib import Path

def save_results(self, results_df: pd.DataFrame):
    output_dir = Path(self.config.output_dir)
    ensure_dir(output_dir)  # Garante que diretório existe
    
    output_path = output_dir / "results.csv"
    results_df.to_csv(output_path, index=False)
    _log.info(f"Saved results to: {output_path}")
```