# scripts/run_retrieval.py
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
    p.add_argument("--sem-dim", type=int, default=384)  # apenas para stub

    # tf-idf
    p.add_argument("--tfidf-backend", type=str, choices=["sklearn","pyserini"], default="sklearn")
    p.add_argument("--tfidf-dim", type=int, default=1000)

    # entidades
    p.add_argument("--graph-model", type=str, default="BAAI/bge-large-en-v1.5")
    p.add_argument("--ner-backend", dest="ner_backend", type=str, choices=["scispacy","spacy","none"], default="scispacy")
    p.add_argument("--ner-model", dest="ner_model", type=str, default="")
    p.add_argument("--ner-allowed-labels", dest="ner_allowed_labels", type=str, default="",
                   help='CSV de labels aceitas (ex.: "DISEASE,CHEMICAL"). Vazio = sem filtro.')
    p.add_argument("--ner-use-noun-chunks", dest="ner_use_noun_chunks", action="store_true", default=True)
    p.add_argument("--ner-batch-size", dest="ner_batch_size", type=int, default=64)
    p.add_argument("--ner-n-process", dest="ner_n_process", type=int, default=1)
    p.add_argument("--entity-artifacts", dest="entity_artifact_dir", type=str, default="",
                   help="Pasta para cachear IDF/embeddings de entidade")
    p.add_argument("--entity-force-rebuild", dest="entity_force_rebuild", action="store_true", default=False)

    # dispositivo
    p.add_argument("--device", type=str, default=None, help="ex.: 'cuda:0' ou 'cpu'")

    # FAISS
    p.add_argument("--faiss-factory", dest="faiss_factory", type=str, default="",
                   help='Ex.: "OPQ64,IVF4096,PQ64x8". Vazio = FlatIP')
    p.add_argument("--faiss-metric", dest="faiss_metric", type=str, choices=["ip","l2"], default="ip")
    p.add_argument("--faiss-nprobe", dest="faiss_nprobe", type=int, default=0)
    p.add_argument("--faiss-train-size", dest="faiss_train_size", type=int, default=0,
                   help="0 = auto (>= 30*nlist).")
    p.add_argument("--index-artifacts", dest="index_artifact_dir", type=str, default="",
                   help="Pasta para salvar/carregar índice FAISS")
    p.add_argument("--index-name", dest="index_name", type=str, default="hybrid.index")
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
        allowed_labels = [s.strip() for s in args.ner_allowed_labels.split(",") if s.strip()] if args.ner_allowed_labels else None
        retr = HybridRetriever(
            sem_dim=args.sem_dim,
            tfidf_dim=args.tfidf_dim,
            semantic_backend=args.semantic_backend,
            semantic_model_name=args.semantic_model,
            tfidf_backend=args.tfidf_backend,
            query_prefix=args.query_prefix,
            doc_prefix=args.doc_prefix,
            graph_model_name=args.graph_model,
            ner_backend=args.ner_backend,
            ner_model=(args.ner_model or None),
            ner_use_noun_chunks=args.ner_use_noun_chunks,
            ner_batch_size=args.ner_batch_size,
            ner_n_process=args.ner_n_process,
            ner_allowed_labels=allowed_labels,
            entity_artifact_dir=(args.entity_artifact_dir or None),
            entity_force_rebuild=args.entity_force_rebuild,
            device=args.device,
            faiss_factory=(args.faiss_factory or None),
            faiss_metric=args.faiss_metric,
            faiss_nprobe=(args.faiss_nprobe or None if args.faiss_nprobe <= 0 else args.faiss_nprobe),
            faiss_train_size=args.faiss_train_size,
            index_artifact_dir=(args.index_artifact_dir or None),
            index_name=args.index_name,
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
