from __future__ import annotations
import argparse
from pathlib import Path
from src.datasets.loader import load_beir_dataset, select_split, as_documents
from src.retrievers.bm25_basic import BM25Basic
from src.retrievers.dense_faiss import DenseFaissStub
from src.retrievers.hybrid_faiss import HybridRetriever
from src.utils.logging import get_logger

RETRIEVERS = {
    "bm25": BM25Basic,
    "dense": DenseFaissStub,
    "hybrid": HybridRetriever,
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True, help="Pasta com corpus/queries/qrels (BEIR processado)")
    p.add_argument("--retriever", type=str, choices=RETRIEVERS.keys(), default="hybrid")
    p.add_argument("--sem-dim", type=int, default=384)
    p.add_argument("--tfidf-dim", type=int, default=1000)
    return p.parse_args()

def main():
    args = parse_args()
    log = get_logger()
    root = Path(args.dataset_root)
    corpus, queries, qrels = load_beir_dataset(root)
    # build docs
    docs = as_documents(corpus)

    # instância do retriever
    if args.retriever == "hybrid":
        retr = HybridRetriever(sem_dim=args.sem_dim, tfidf_dim=args.tfidf_dim)
    elif args.retriever == "dense":
        retr = DenseFaissStub(dim=args.sem_dim)
    else:
        retr = BM25Basic()

    log.info(f"Construindo índice para {args.retriever} em {root} ...")
    retr.build_index(docs)
    log.info("OK. (Stub não salva cache em disco ainda; adicionaremos depois)")

if __name__ == "__main__":
    main()
