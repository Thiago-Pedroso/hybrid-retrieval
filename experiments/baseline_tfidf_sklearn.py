#!/usr/bin/env python3
"""
Baseline TF-IDF Evaluation Script (Pure sklearn)
Avalia o desempenho do TF-IDF puro (sklearn) nos datasets BEIR: scifact, fiqa, nfcorpus
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Configurações
ROOT = Path("./data").resolve()
DATASETS = ["scifact", "fiqa", "nfcorpus"]
TOPK = 100

import logging

# Desabilita todos os loggers conhecidos
logging.disable(logging.CRITICAL)
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).disabled = True


def load_parquet_or_jsonl(path_parquet: Path, path_jsonl: Path) -> pd.DataFrame:
    if path_parquet.exists():
        return pd.read_parquet(path_parquet)
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


# Tokenização simples
_tok_re = re.compile(r"[A-Za-z0-9_]+")
def tokenize(text: str):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return [t.lower() for t in _tok_re.findall(text)]


def build_tfidf_index(df_corpus: pd.DataFrame):
    """Constrói índice TF-IDF"""
    # concatena título + texto
    texts = (df_corpus["title"].fillna("") + " " + df_corpus["text"].fillna("")).tolist()
    
    # Cria vetorizador TF-IDF com tokenizer customizado
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        lowercase=False,  # já fazemos lowercase no tokenizer
        token_pattern=None  # usamos tokenizer customizado
    )
    
    # Fit e transforma o corpus
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape} (docs × features)")
    
    return vectorizer, tfidf_matrix


def rank_with_tfidf(vectorizer, tfidf_matrix, query_text: str, topk=TOPK):
    """Ranking usando TF-IDF e similaridade de cosseno"""
    # Transforma a query usando o mesmo vetorizador
    query_vec = vectorizer.transform([query_text])
    
    # Calcula similaridade de cosseno
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Pega top-k índices
    top_idx = np.argpartition(scores, -topk)[-topk:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]  # ordena por score desc
    
    return top_idx, scores[top_idx]


def evaluate_dataset(name: str, topk=TOPK, only_test=True):
    """Avalia um dataset e retorna as métricas"""
    print(f"\n====== {name} ======")
    df_corpus, df_queries, df_qrels = load_beir_processed(name)
    print("corpus:", df_corpus.shape, "| queries:", df_queries.shape, "| qrels:", df_qrels.shape)

    desired = "test" if only_test else "test"
    split = pick_split_available(df_qrels, prefer=desired)
    if only_test and split != "test":
        print(f"⚠️ Split 'test' não encontrado em {name}; usando '{split}' para rodar mesmo assim.")
    qrels_split = df_qrels[df_qrels["split"] == split].copy()
    print("usando split:", split, "| qrels:", qrels_split.shape)

    # mapa query_id -> {doc_id: score}
    qrels_map = {}
    for row in qrels_split.itertuples(index=False):
        qrels_map.setdefault(row.query_id, {})[row.doc_id] = int(getattr(row, "score", 1))

    # constrói índice lexical
    vectorizer, tfidf_matrix = build_tfidf_index(df_corpus)

    # prepara lookup de queries existentes
    qdf = df_queries[df_queries["query_id"].isin(qrels_split["query_id"].unique())]
    q_lookup = dict(zip(qdf["query_id"], qdf["query"]))

    metrics = {"MRR@10": [], "nDCG@10": [], "MAP@10": [], "Recall@10": []}

    for qid, gold_gains in tqdm(qrels_map.items(), desc=f"TFIDF {name} ({split})"):
        qtxt = q_lookup.get(qid)
        if qtxt is None:
            continue
        gold_set = {d for d, s in gold_gains.items() if s > 0}
        top_idx, _ = rank_with_tfidf(vectorizer, tfidf_matrix, qtxt, topk=topk)
        ranked = df_corpus["doc_id"].iloc[top_idx].tolist()  # converte índices para doc_ids

        metrics["MRR@10"].append(mrr_at_k(ranked, gold_set, k=10))
        metrics["nDCG@10"].append(ndcg_at_k(ranked, gold_gains, k=10))
        metrics["MAP@10"].append(average_precision_at_k(ranked, gold_set, k=10))
        metrics["Recall@10"].append(recall_at_k(ranked, gold_set, k=10))

    # Calcula médias
    results = {m: float(np.mean(v)) if v else 0.0 for m, v in metrics.items()}
    
    print("Resultados (médias):", results)
    
    return results


def main():
    """Executa avaliação em todos os datasets"""
    all_results = {}
    for ds in DATASETS:
        all_results[ds] = evaluate_dataset(ds, only_test=True)

    print("\n== Resumo (médias por dataset) ==")
    df_results = pd.DataFrame(all_results).T
    print(df_results.to_string())
    
    return all_results


if __name__ == "__main__":
    main()

