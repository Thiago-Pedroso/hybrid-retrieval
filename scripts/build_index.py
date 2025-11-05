from __future__ import annotations
import argparse
from pathlib import Path
from src.datasets.loader import load_beir_dataset, select_split, as_documents
from src.retrievers.bm25_basic import BM25Basic
from src.retrievers.dense_faiss import DenseFaiss
from src.retrievers.hybrid_faiss import HybridRetriever
from src.vectorizers.factory import create_vectorizer
from src.indexes.factory import create_index
from src.fusion.factory import create_weight_policy, create_reranker
from src.utils.logging import get_logger

RETRIEVERS = {
    "bm25": BM25Basic,
    "dense": DenseFaiss,
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
        # Create components using new modular API
        vectorizer = create_vectorizer({
            "type": "tri_modal",
            "semantic": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "tfidf": {"dim": args.tfidf_dim},
            "graph": {"model": "BAAI/bge-large-en-v1.5"},
        })
        index = create_index({"type": "faiss", "metric": "ip"}, vectorizer)
        weight_policy = create_weight_policy({"policy": "heuristic"})
        reranker = create_reranker("tri_modal", vectorizer)
        retr = HybridRetriever(
            vectorizer=vectorizer,
            index=index,
            reranker=reranker,
            weight_policy=weight_policy,
            topk_first=150,
        )
    elif args.retriever == "dense":
        # Map dimension to model (384 -> MiniLM, 768 -> mpnet, 1024 -> BGE-large)
        model_map = {
            384: "sentence-transformers/all-MiniLM-L6-v2",
            768: "sentence-transformers/all-mpnet-base-v2",
            1024: "BAAI/bge-large-en-v1.5",
        }
        model_name = model_map.get(args.sem_dim, "sentence-transformers/all-MiniLM-L6-v2")
        retr = DenseFaiss(model_name=model_name)
    else:
        retr = BM25Basic()

    log.info(f"Construindo índice para {args.retriever} em {root} ...")
    retr.build_index(docs)
    log.info("OK. (Stub não salva cache em disco ainda; adicionaremos depois)")

if __name__ == "__main__":
    main()
