from pathlib import Path
import pandas as pd
from typing import Tuple, List
from .schema import Document, Query

def _load_parquet_or_jsonl(path_parquet: Path, path_jsonl: Path) -> pd.DataFrame:
    """
    Prefere Parquet; se faltar engine (pyarrow/fastparquet) ou der erro, cai para JSONL.
    """
    if path_parquet.exists():
        try:
            # tenta engine explícita; se não tiver, pandas pode levantar ImportError/ValueError
            return pd.read_parquet(path_parquet, engine="pyarrow")
        except Exception as e_parq:
            print(f"[loader] Aviso: falha ao ler Parquet ({e_parq}). "
                  f"Tentando JSONL em {path_jsonl} ...")
    if path_jsonl.exists():
        return pd.read_json(path_jsonl, lines=True)
    raise FileNotFoundError(f"Arquivos não encontrados: {path_parquet} | {path_jsonl}")

def load_beir_dataset(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega BEIR processado (corpus/queries/qrels)."""
    corpus = _load_parquet_or_jsonl(root / "corpus.parquet", root / "corpus.jsonl")
    queries = _load_parquet_or_jsonl(root / "queries.parquet", root / "queries.jsonl")
    qrels   = _load_parquet_or_jsonl(root / "qrels.parquet",   root / "qrels.jsonl")
    # normalizações leves
    corpus["doc_id"] = corpus["doc_id"].astype(str)
    queries["query_id"] = queries["query_id"].astype(str)
    qrels["doc_id"] = qrels["doc_id"].astype(str)
    qrels["query_id"] = qrels["query_id"].astype(str)
    if "split" not in qrels.columns:
        qrels["split"] = "test"
    if "score" not in qrels.columns:
        qrels["score"] = 1
    return corpus, queries, qrels

def select_split(qrels: pd.DataFrame, prefer=("test","dev","validation","train")) -> str:
    present = set(qrels["split"].unique().tolist())
    for s in prefer:
        if s in present:
            return s
    return qrels["split"].iloc[0]

def as_documents(corpus: pd.DataFrame) -> List[Document]:
    rows = []
    for r in corpus.itertuples(index=False):
        rows.append(Document(getattr(r, "doc_id"), getattr(r, "title", None), getattr(r, "text", ""), getattr(r, "metadata", None)))
    return rows

def as_queries(queries: pd.DataFrame) -> List[Query]:
    return [Query(qid, txt) for qid, txt in zip(queries["query_id"], queries["query"])]
