from __future__ import annotations
import argparse
from pathlib import Path
from src.datasets.loader import load_beir_dataset, select_split, as_documents, as_queries
from src.retrievers.bm25_basic import BM25Basic
from src.retrievers.dense_faiss import DenseFaissStub
from src.retrievers.hybrid_faiss import HybridRetriever
from src.utils.io import ensure_dir, predictions_to_jsonl, write_jsonl
from src.utils.logging import get_logger

RETRIEVERS = ("bm25", "dense", "hybrid")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--retriever", type=str, choices=RETRIEVERS, default="hybrid")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", type=str, default="./outputs/predictions.jsonl")
    p.add_argument("--split-prefer", type=str, default="test,dev,validation,train")

    # semântico
    p.add_argument("--semantic-backend", type=str, choices=["hf","stub"], default="hf")
    p.add_argument("--semantic-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--query-prefix", type=str, default="")
    p.add_argument("--doc-prefix", type=str, default="")
    p.add_argument("--sem-dim", type=int, default=384)  # só para stub

    # tf-idf
    p.add_argument("--tfidf-backend", type=str, choices=["sklearn","pyserini"], default="sklearn")
    p.add_argument("--tfidf-dim", type=int, default=1000)

    # entidades (slice g)
    p.add_argument("--graph-model", type=str, default="BAAI/bge-large-en-v1.5")
    p.add_argument("--ner-backend", type=str, choices=["scispacy","spacy"], default="scispacy")
    p.add_argument("--ner-model", type=str, default="")
    p.add_argument("--ner-use-noun-chunks", action="store_true", default=True)
    p.add_argument("--ner-batch-size", type=int, default=64)
    p.add_argument("--ner-n-process", type=int, default=1)

    # dispositivo
    p.add_argument("--device", type=str, default=None, help="ex.: 'cuda:0' ou 'cpu'")
    return p.parse_args()

def main():
    args = parse_args()
    log = get_logger()
    root = Path(args.dataset_root)

    corpus, queries, qrels = load_beir_dataset(root)
    split = select_split(qrels, tuple(x.strip() for x in args.split_prefer.split(",")))
    qrels_split = qrels[qrels["split"] == split].copy()
    log.info(f"Usando split={split} | queries no qrels: {qrels_split['query_id'].nunique()}")

    docs = as_documents(corpus)
    qids = set(qrels_split["query_id"].unique().tolist())
    queries = queries[queries["query_id"].isin(qids)]
    qlist = as_queries(queries)

    if args.retriever == "hybrid":
        retr = HybridRetriever(
            sem_dim=args.sem_dim,
            tfidf_dim=args.tfidf_dim,
            semantic_backend=args.semantic_backend,
            semantic_model_name=args.semantic_model,
            tfidf_backend=args.tfidf_backend,
            query_prefix=args.query_prefix,
            doc_prefix=args.doc_prefix,
            graph_model_name=args.graph_model,
            ner_backend=args.ner-backend if hasattr(args, "ner-backend") else args.ner_backend,  # robustness
            ner_model=(args.ner_model or None),
            ner_use_noun_chunks=args.ner_use_noun_chunks,
            ner_batch_size=args.ner_batch_size,
            ner_n_process=args.ner_n_process,
            device=args.device,
        )
    elif args.retriever == "dense":
        retr = DenseFaissStub(dim=args.sem_dim)
    else:
        retr = BM25Basic()

    log.info(f"Construindo índice para {args.retriever} ...")
    retr.build_index(docs)
    log.info(f"Recuperando top-{args.k} para {len(qlist)} queries ...")
    preds = retr.retrieve(qlist, k=args.k)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    write_jsonl(out_path, predictions_to_jsonl(preds))
    log.info(f"Predições salvas em: {out_path}")

if __name__ == "__main__":
    main()
