# scripts/run_retrieval.py
from __future__ import annotations
import argparse
from pathlib import Path
from src.datasets.loader import load_beir_dataset, select_split, as_documents, as_queries
from src.retrievers.bm25_basic import BM25Basic
from src.retrievers.dense_faiss import DenseFaiss
from src.retrievers.hybrid_faiss import HybridRetriever
from src.vectorizers.factory import create_vectorizer
from src.indexes.factory import create_index
from src.fusion.factory import create_weight_policy, create_reranker
from src.utils.io import ensure_dir, predictions_to_jsonl, write_jsonl
from src.utils.logging import get_logger, set_log_level, enable_file_logging, log_time

RETRIEVERS = ("bm25", "dense", "hybrid")

def parse_args():
    p = argparse.ArgumentParser(description="Sistema de Retrieval Híbrido")
    p.add_argument("--dataset-root", type=str, required=True, help="Pasta com corpus/queries/qrels (BEIR processado)")
    p.add_argument("--retriever", type=str, choices=RETRIEVERS, default="hybrid", help="Tipo de retriever")
    p.add_argument("--k", type=int, default=10, help="Top-K resultados por query")
    p.add_argument("--out", type=str, default="./outputs/predictions.jsonl", help="Arquivo de saída")
    p.add_argument("--split-prefer", type=str, default="test,dev,validation,train", help="Ordem de preferência de splits")
    
    # Logging
    p.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                   default="INFO", help="Nível de log (também via HYBRID_LOG_LEVEL env var)")
    p.add_argument("--log-file", type=str, default=None, 
                   help="Se fornecido, salva logs em arquivo (ex: ./logs/run.log)")

    # semântico
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
    p.add_argument("--ner-batch-size", dest="ner_batch_size", type=int, default=128)
    p.add_argument("--ner-n-process", dest="ner_n_process", type=int, default=4)
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
    
    # Configura logging
    set_log_level(args.log_level)
    if args.log_file:
        log_path = Path(args.log_file)
        enable_file_logging(log_path.parent, log_path.name)
    
    log = get_logger("main")
    log.info("="*80)
    log.info("🚀 SISTEMA DE RETRIEVAL HÍBRIDO")
    log.info("="*80)
    
    root = Path(args.dataset_root)
    log.info(f"📂 Dataset root: {root}")

    with log_time(log, "Carregando dataset"):
        corpus, queries, qrels = load_beir_dataset(root)
    
    log.info(f"📊 Dataset: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")
    
    split = select_split(qrels, tuple(x.strip() for x in args.split_prefer.split(",")))
    qrels_split = qrels[qrels["split"] == split].copy()
    n_queries_in_split = qrels_split['query_id'].nunique()
    log.info(f"🎯 Usando split={split} | {n_queries_in_split} queries no qrels")

    docs = as_documents(corpus)
    qids = set(qrels_split["query_id"].unique().tolist())
    queries = queries[queries["query_id"].isin(qids)]
    qlist = as_queries(queries)
    log.info(f"✓ Preparado: {len(docs)} documentos, {len(qlist)} queries")

    if args.retriever == "hybrid":
        # Create components using new modular API
        allowed_labels = [s.strip() for s in args.ner_allowed_labels.split(",") if s.strip()] if args.ner_allowed_labels else None
        vectorizer_config = {
            "type": "tri_modal",
            "semantic": {
                "model": args.semantic_model,
                "query_prefix": args.query_prefix,
                "doc_prefix": args.doc_prefix,
            },
            "tfidf": {
                "dim": args.tfidf_dim,
                "backend": args.tfidf_backend,
            },
            "graph": {
                "model": args.graph_model,
                "ner_backend": args.ner_backend,
                "ner_model": args.ner_model or None,
                "ner_use_noun_chunks": args.ner_use_noun_chunks,
                "ner_batch_size": args.ner_batch_size,
                "ner_n_process": args.ner_n_process,
                "ner_allowed_labels": allowed_labels,
                "entity_artifact_dir": args.entity_artifact_dir or None,
                "entity_force_rebuild": args.entity_force_rebuild,
            },
        }
        if args.device:
            vectorizer_config["semantic"]["device"] = args.device
        
        vectorizer = create_vectorizer(vectorizer_config)
        
        index_config = {
            "type": "faiss",
            "factory": args.faiss_factory or None,
            "metric": args.faiss_metric,
            "nprobe": args.faiss_nprobe if args.faiss_nprobe > 0 else None,
            "train_size": args.faiss_train_size,
            "artifact_dir": args.index_artifact_dir or None,
            "index_name": args.index_name,
        }
        index = create_index(index_config, vectorizer)
        
        weight_policy = create_weight_policy({"policy": args.policy})
        reranker = create_reranker("tri_modal", vectorizer)
        
        retr = HybridRetriever(
            vectorizer=vectorizer,
            index=index,
            reranker=reranker,
            weight_policy=weight_policy,
            topk_first=args.topk_first,
        )
    elif args.retriever == "dense":
        # Map dimension to model (384 -> MiniLM, 768 -> mpnet, 1024 -> BGE-large)
        model_map = {
            384: "sentence-transformers/all-MiniLM-L6-v2",
            768: "sentence-transformers/all-mpnet-base-v2",
            1024: "BAAI/bge-large-en-v1.5",
        }
        model_name = model_map.get(args.sem_dim, "sentence-transformers/all-MiniLM-L6-v2")
        retr = DenseFaiss(
            model_name=model_name,
            query_prefix=args.query_prefix,
            doc_prefix=args.doc_prefix,
        )
    else:
        retr = BM25Basic()

    log.info(f"🔧 Retriever: {args.retriever}")
    
    with log_time(log, f"Build index ({args.retriever})"):
        retr.build_index(docs)
    
    with log_time(log, f"Retrieve top-{args.k}"):
        preds = retr.retrieve(qlist, k=args.k)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    with log_time(log, "Salvando predições"):
        write_jsonl(out_path, predictions_to_jsonl(preds))
    
    log.info("="*80)
    log.info(f"✅ CONCLUÍDO! Predições salvas em: {out_path}")
    log.info("="*80)

if __name__ == "__main__":
    main()
