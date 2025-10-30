# Benchmark pré-rerank (Experimento A) — Tri-modal (s + t + g)

Este guia mostra como rodar o pipeline sem reranking (FAISS apenas), com split 80/10/10 determinístico por query.

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU (opcional):
- Passe `--device cuda:0` nos scripts para usar GPU NVIDIA.
- Em macOS, MPS é habilitado automaticamente; em Linux/Windows não há restrições de threads.

## 2) Download dos datasets (BEIR)

```bash
python download_datasets.py --datasets scifact,fiqa,nfcorpus --root ./data --format jsonl
```

Estrutura esperada:
- `data/<dataset>/processed/beir/{corpus.jsonl, queries.jsonl, qrels.jsonl}`

## 3) Split 80/10/10 (determinístico)

Simular e materializar qrels 80/10/10 por query_id (seed=42):

```bash
# SciFact
python scripts/check_dataset_splits.py --dataset scifact --simulate-801010 --write-split \
  --out ./data/scifact/processed/beir/qrels_801010.jsonl

# FIQA
python scripts/check_dataset_splits.py --dataset fiqa --simulate-801010 --write-split \
  --out ./data/fiqa/processed/beir/qrels_801010.jsonl

# NFCorpus
python scripts/check_dataset_splits.py --dataset nfcorpus --simulate-801010 --write-split \
  --out ./data/nfcorpus/processed/beir/qrels_801010.jsonl
```

## 4) Execução do Experimento A (sem rerank)

Use o script A/B, que roda apenas a fase FAISS (pré-rerank), e injete o qrels 80/10/10:

```bash
python scripts/run_prerank_ab.py \
  --dataset scifact \
  --k 10 \
  --semantic-a sentence-transformers/all-MiniLM-L6-v2 \
  --semantic-b BAAI/bge-large-en-v1.5 \
  --graph-model BAAI/bge-large-en-v1.5 \
  --qrels-path ./data/scifact/processed/beir/qrels_801010.jsonl \
  --device cuda:0 \
  --csv-out ./outputs/ab_scifact_801010.csv
```

Notas:
- O ranking utilizado para as métricas é exatamente o resultado do FAISS (sem reordenação posterior).
- Similaridade: produto interno sobre vetores L2-normalizados (equivalente a cosseno).
- s, t, g são normalizados individualmente; depois concatenados e normalizados novamente.

## 5) Avaliação independente (opcional)

Se tiver gerado `predictions.jsonl` com outro script, avalie assim:

```bash
python scripts/evaluate.py \
  --dataset-root ./data/scifact/processed/beir \
  --predictions ./outputs/predictions.jsonl \
  --qrels-path ./data/scifact/processed/beir/qrels_801010.jsonl \
  --ks "1,3,5,10" \
  --out ./outputs/metrics_801010.csv
```