# Datasets

Este módulo fornece funcionalidades para carregar e manipular datasets de recuperação de informação no formato BEIR.

## 📚 Sobre os Datasets

Este projeto utiliza datasets do **BEIR (Benchmarking IR)**, um benchmark heterogêneo para avaliação de sistemas de recuperação de informação. Os datasets são fornecidos no formato padronizado que inclui corpus de documentos, queries (consultas) e qrels (relevance judgments).

### Datasets Suportados

#### 1. **SciFact**
- **Descrição**: Dataset de verificação de declarações científicas baseado em artigos acadêmicos
- **Domínio**: Ciência biomédica
- **Tamanho**: 
  - ~5.000 documentos (abstracts de artigos científicos)
  - ~300 queries (claims científicas para verificação)
- **Composição**:
  - **Corpus**: Abstracts de artigos científicos com título e texto
  - **Queries**: Declarações científicas que precisam ser verificadas
  - **Qrels**: Julgamentos de relevância indicando quais documentos suportam/refutam cada claim
- **Splits disponíveis**: train, dev, test
- **Fonte Original**: AllenAI (https://scifact.s3-us-west-2.amazonaws.com)
- **Versão BEIR**: Formato padronizado do benchmark BEIR
- **Caso de Uso**: Fact-checking científico, verificação de claims biomédicas

#### 2. **FIQA**
- **Descrição**: Financial Opinion Mining and Question Answering
- **Domínio**: Finanças (posts de fóruns financeiros e sites de Q&A)
- **Tamanho**:
  - ~57.000 documentos (posts e respostas de fóruns financeiros)
  - ~6.600 queries (perguntas financeiras)
- **Composição**:
  - **Corpus**: Posts de comunidades financeiras (Stack Exchange, Reddit, etc.)
  - **Queries**: Perguntas sobre investimentos, finanças pessoais, mercados
  - **Qrels**: Julgamentos de relevância baseados em upvotes e aceitação de respostas
- **Splits disponíveis**: train, dev, test
- **Caso de Uso**: Question answering em domínio financeiro, busca de opiniões especializadas

#### 3. **NFCorpus**
- **Descrição**: Nutrition Facts Corpus - Linking PubMed articles to NutritionFacts.org articles
- **Domínio**: Nutrição e medicina
- **Tamanho**:
  - ~3.600 documentos (artigos médicos do PubMed)
  - ~323 queries (títulos de artigos do NutritionFacts.org)
- **Composição**:
  - **Corpus**: Abstracts de artigos médicos sobre nutrição
  - **Queries**: Consultas baseadas em artigos de divulgação científica
  - **Qrels**: Links entre artigos científicos e divulgação
- **Splits disponíveis**: train, dev, test
- **Caso de Uso**: Recuperação de evidências científicas para artigos de divulgação, cross-domain retrieval

### Formato BEIR Padronizado

Todos os datasets seguem o formato BEIR com três componentes principais:

1. **corpus.{parquet|jsonl}**: Documentos indexáveis
   - `doc_id`: Identificador único do documento
   - `title`: Título do documento (opcional)
   - `text`: Conteúdo textual principal
   - `metadata`: Metadados adicionais (opcional)

2. **queries.{parquet|jsonl}**: Consultas de busca
   - `query_id`: Identificador único da query
   - `query`: Texto da consulta

3. **qrels.{parquet|jsonl}**: Julgamentos de relevância (ground truth)
   - `query_id`: ID da query
   - `doc_id`: ID do documento
   - `score`: Grau de relevância (tipicamente 0 ou 1, mas pode variar)
   - `split`: Partição do dataset (train/dev/test)

## 🚀 Preparação dos Dados

### Pré-requisitos

Antes de usar os datasets neste formato, você precisa executar o script de download:

```bash
# Instalar dependências necessárias
pip install pandas pyarrow requests tqdm

# Para SciFact via Hugging Face (opcional):
pip install datasets
```

### Download e Processamento

Execute o script `download_datasets.py` na raiz do projeto:

```bash
# Baixar todos os datasets (SciFact, FIQA, NFCorpus)
python download_datasets.py \
  --datasets scifact,fiqa,nfcorpus \
  --root ./data \
  --format parquet

# Baixar apenas SciFact
python download_datasets.py \
  --datasets scifact \
  --root ./data \
  --format parquet

# Usar Hugging Face como fonte para SciFact
python download_datasets.py \
  --datasets scifact \
  --root ./data \
  --format parquet \
  --scifact-source hf
```

### Estrutura de Diretórios Criada

Após executar o script de download, a seguinte estrutura será criada:

```
data/
├── scifact/
│   ├── raw/
│   │   ├── original/         # Formato original do AllenAI
│   │   │   └── data.tar.gz
│   │   └── beir/             # Formato BEIR
│   │       ├── scifact.zip
│   │       └── extracted/
│   └── processed/
│       ├── original/         # Normalizado em parquet
│       │   ├── corpus.parquet
│       │   ├── claims_train.parquet
│       │   ├── claims_dev.parquet
│       │   └── claims_test.parquet
│       └── beir/             # Normalizado em parquet
│           ├── corpus.parquet
│           ├── queries.parquet
│           └── qrels.parquet
├── fiqa/
│   ├── raw/beir/
│   │   ├── fiqa.zip
│   │   └── extracted/
│   └── processed/beir/
│       ├── corpus.parquet
│       ├── queries.parquet
│       └── qrels.parquet
└── nfcorpus/
    ├── raw/beir/
    │   ├── nfcorpus.zip
    │   └── extracted/
    └── processed/beir/
        ├── corpus.parquet
        ├── queries.parquet
        └── qrels.parquet
```

## 🔧 Uso do Módulo

### Funcionalidades do Código

Este módulo (`src/datasets/`) fornece duas funcionalidades principais:

#### 1. **Schema de Dados** (`schema.py`)

Define estruturas de dados tipadas para documentos e queries:

```python
from src.datasets.schema import Document, Query

# Document: representa um documento do corpus
doc = Document(
    doc_id="12345",
    title="Título do documento",
    text="Conteúdo do documento...",
    metadata={"author": "João Silva"}
)

# Query: representa uma consulta
query = Query(
    query_id="q1",
    text="Como investir em ações?"
)
```

#### 2. **Carregamento de Dados** (`loader.py`)

Funções para carregar datasets BEIR processados:

##### `load_beir_dataset(root: Path)`

Carrega os três componentes de um dataset BEIR:

```python
from pathlib import Path
from src.datasets.loader import load_beir_dataset

# Carregar dataset SciFact
root = Path("data/scifact/processed/beir")
corpus, queries, qrels = load_beir_dataset(root)

# Retorna 3 DataFrames do pandas:
# - corpus: DataFrame com colunas [doc_id, title, text, metadata]
# - queries: DataFrame com colunas [query_id, query]
# - qrels: DataFrame com colunas [query_id, doc_id, score, split]
```

**Características**:
- Suporta tanto **Parquet** (preferencial) quanto **JSONL** (fallback)
- Normaliza tipos de dados (IDs como strings)
- Adiciona coluna `split` padrão ("test") se ausente
- Adiciona coluna `score` padrão (1) se ausente
- Tratamento robusto de erros com fallback automático

##### `select_split(qrels: DataFrame, prefer=("test","dev","validation","train"))`

Seleciona automaticamente o split mais apropriado:

```python
from src.datasets.loader import select_split

# Selecionar split preferencial
split_name = select_split(qrels)  # Retorna "test" se disponível
split_name = select_split(qrels, prefer=("dev", "test"))  # Prioriza "dev"

# Filtrar qrels pelo split selecionado
qrels_split = qrels[qrels["split"] == split_name]
```

##### `as_documents(corpus: DataFrame)` e `as_queries(queries: DataFrame)`

Converte DataFrames para listas de objetos tipados:

```python
from src.datasets.loader import as_documents, as_queries

# Converter para objetos Document
docs = as_documents(corpus)  # List[Document]

# Converter para objetos Query
qs = as_queries(queries)  # List[Query]

# Usar em código tipado
for doc in docs:
    print(f"{doc.doc_id}: {doc.title}")
```


## 📊 Estatísticas dos Datasets

| Dataset   | Documentos | Queries (test) | Avg Doc Length | Avg Query Length | Domínio |
|-----------|------------|----------------|----------------|------------------|---------|
| SciFact   | ~5K        | ~300           | ~200 tokens    | ~20 tokens       | Ciência |
| FIQA      | ~57K       | ~6.6K          | ~130 tokens    | ~11 tokens       | Finanças|
| NFCorpus  | ~3.6K      | ~323           | ~250 tokens    | ~3 tokens        | Nutrição|


## 📖 Referências

- **BEIR Benchmark**: Thakur, N., et al. (2021). "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models"
- **SciFact**: Wadden, D., et al. (2020). "Fact or Fiction: Verifying Scientific Claims"
- **FIQA**: Maia, M., et al. (2018). "WWW'18 Open Challenge: Financial Opinion Mining and Question Answering"
- **NFCorpus**: Boteva, V., et al. (2016). "A Full-Text Learning to Rank Dataset for Medical Information Retrieval"

