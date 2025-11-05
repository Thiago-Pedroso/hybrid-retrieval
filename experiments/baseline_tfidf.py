#!/usr/bin/env python3
"""
Baseline TF-IDF Evaluation Script
Avalia o desempenho do TF-IDF nos datasets BEIR: scifact, fiqa, nfcorpus
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.datasets.schema import Document, Query
from src.retrievers.tfidf_faiss import TFIDFRetriever

import logging

# Desabilita todos os loggers conhecidos
logging.disable(logging.CRITICAL)
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).disabled = True


# Configurações
ROOT = Path("./data").resolve()
DATASETS = ["scifact", "fiqa", "nfcorpus"]
TOPK = 100


def load_parquet_or_jsonl(path_parquet: Path, path_jsonl: Path) -> pd.DataFrame:
    if path_parquet.exists():
        try:
            return pd.read_parquet(path_parquet)
        except ImportError:
            pass  # pyarrow não disponível, tenta jsonl
    if path_jsonl.exists():
        return pd.read_json(path_jsonl, lines=True)
    raise FileNotFoundError(f"Faltam arquivos: {path_parquet} | {path_jsonl}")


def load_beir_processed(ds_name: str):
    base = ROOT / ds_name / "processed" / "beir"
    paths = {
        "corpus":  (base / "corpus.parquet",  base / "corpus.jsonl"),
        "queries": (base / "queries.parquet", base / "queries.jsonl"),
        "qrels":   (base / "qrels.parquet",   base / "qrels.jsonl"),
    }
    df_corpus  = load_parquet_or_jsonl(*paths["corpus"])
    df_queries = load_parquet_or_jsonl(*paths["queries"])
    df_qrels   = load_parquet_or_jsonl(*paths["qrels"])

    # Normalizações
    df_corpus["doc_id"]   = df_corpus["doc_id"].astype(str)
    df_queries["query_id"] = df_queries["query_id"].astype(str)
    if "split" not in df_qrels.columns:
        df_qrels["split"] = "test"
    df_qrels["query_id"] = df_qrels["query_id"].astype(str)
    df_qrels["doc_id"]   = df_qrels["doc_id"].astype(str)
    if "score" not in df_qrels.columns:
        df_qrels["score"] = 1

    return df_corpus, df_queries, df_qrels


def pick_split_available(qrels: pd.DataFrame, prefer="test"):
    order = [prefer, "dev", "validation", "train"]
    present = set(qrels["split"].unique().tolist())
    for s in order:
        if s in present:
            return s
    return qrels["split"].iloc[0]


# Métricas
def mrr_at_k(ranked, gold_set, k=10):
    for i, did in enumerate(ranked[:k], start=1):
        if did in gold_set:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked, gains, k=10):
    dcg = 0.0
    for i, did in enumerate(ranked[:k], start=1):
        g = gains.get(did, 0.0)
        if g > 0:
            dcg += (2**g - 1) / np.log2(i + 1)
    return dcg


def ndcg_at_k(ranked, gains, k=10):
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = 0.0
    for i, g in enumerate(ideal, start=1):
        idcg += (2**g - 1) / np.log2(i + 1)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked, gains, k) / idcg


def average_precision_at_k(ranked, gold_set, k=10):
    hits, s = 0, 0.0
    for i, did in enumerate(ranked[:k], start=1):
        if did in gold_set:
            hits += 1
            s += hits / i
    return 0.0 if not gold_set else s / min(len(gold_set), k)


def recall_at_k(ranked, gold_set, k=10):
    if not gold_set:
        return 0.0
    return len(set(ranked[:k]) & gold_set) / len(gold_set)


def build_tfidf_index(df_corpus: pd.DataFrame, dataset_name: str):
    """Constrói índice TF-IDF usando TFIDFRetriever"""
    documents = []
    for _, row in df_corpus.iterrows():
        doc = Document(
            doc_id=str(row["doc_id"]),
            title=str(row.get("title", "") or ""),
            text=str(row.get("text", "") or "")
        )
        documents.append(doc)
    
    artifact_dir = f"./outputs/artifacts/{dataset_name}_baseline_tfidf"
    retriever = TFIDFRetriever(
        dim=None,  # sem limitação de vocabulário (igual ao notebook)
        min_df=1,  # padrão do sklearn (igual ao notebook)
        backend="sklearn",
        use_faiss=True,  # FAISS habilitado
        artifact_dir=artifact_dir,
        index_name="tfidf.index"
    )
    
    retriever.build_index(documents)
    
    return retriever, documents


def rank_with_tfidf(retriever: TFIDFRetriever, query_text: str, topk=TOPK):
    """Ranking usando TFIDFRetriever"""
    query = Query(query_id="tmp", text=query_text)
    results = retriever.retrieve([query], k=topk)
    ranked_items = results.get("tmp", [])
    doc_ids = [doc_id for doc_id, score in ranked_items]
    scores = [score for doc_id, score in ranked_items]
    
    return doc_ids, scores


def evaluate_dataset(name: str, topk=TOPK):
    """Avalia um dataset e retorna as métricas"""
    df_corpus, df_queries, df_qrels = load_beir_processed(name)
    
    split = pick_split_available(df_qrels, prefer="test")
    qrels_split = df_qrels[df_qrels["split"] == split].copy()
    
    # Mapa query_id -> {doc_id: score}
    qrels_map = {}
    for row in qrels_split.itertuples(index=False):
        qrels_map.setdefault(row.query_id, {})[row.doc_id] = int(getattr(row, "score", 1))

    # Constrói índice TF-IDF
    retriever, documents = build_tfidf_index(df_corpus, name)

    # Prepara lookup de queries
    qdf = df_queries[df_queries["query_id"].isin(qrels_split["query_id"].unique())]
    q_lookup = dict(zip(qdf["query_id"], qdf["query"]))

    metrics = {"MRR@10": [], "nDCG@10": [], "MAP@10": [], "Recall@10": []}

    for qid, gold_gains in tqdm(qrels_map.items(), desc=f"TF-IDF {name} ({split})"):
        qtxt = q_lookup.get(qid)
        if qtxt is None:
            continue
        gold_set = {d for d, s in gold_gains.items() if s > 0}
        
        ranked, _ = rank_with_tfidf(retriever, qtxt, topk=topk)

        metrics["MRR@10"].append(mrr_at_k(ranked, gold_set, k=10))
        metrics["nDCG@10"].append(ndcg_at_k(ranked, gold_gains, k=10))
        metrics["MAP@10"].append(average_precision_at_k(ranked, gold_set, k=10))
        metrics["Recall@10"].append(recall_at_k(ranked, gold_set, k=10))

    # Calcula médias
    results = {m: float(np.mean(v)) if v else 0.0 for m, v in metrics.items()}
    
    return results


def main():
    """Executa avaliação em todos os datasets"""
    print("=" * 60)
    print("Baseline TF-IDF Evaluation")
    print("=" * 60)
    
    all_results = {}
    for ds in DATASETS:
        print(f"\n📊 Avaliando dataset: {ds}")
        all_results[ds] = evaluate_dataset(ds)
    
    print("\n" + "=" * 60)
    print("RESULTADOS FINAIS")
    print("=" * 60)
    
    df_results = pd.DataFrame(all_results).T
    print(df_results.to_string())
    
    print("\n" + "=" * 60)
    print("Média geral:")
    print(df_results.mean().to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()

