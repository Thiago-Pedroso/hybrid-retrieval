"""
Pré-rerank A/B (MiniLM-L6 vs BGE-Large) no índice híbrido (s+t+g), com estatísticas "paper-like" e "BEIR-like".

- "Paper-like" (Tabela I): mostramos Corpus size, Queries (total) e uma coluna chamada Qrels*,
  onde Qrels* == número de QUERIES no split de teste (o que o paper chama de "Qrels").
- "BEIR-like": mostramos também qrels_test_pairs == número de pares (query, doc) relevantes no split de teste.

Exemplos:
  python scripts/run_prerank_ab.py --dataset scifact --k 10
  python scripts/run_prerank_ab.py --dataset scifact --faiss-factory "OPQ64,IVF4096,PQ64x8" --faiss-nprobe 64
  python scripts/run_prerank_ab.py --all --k 10 --csv-out ./outputs/ab_all.csv
"""

from __future__ import annotations

# PROTEÇÃO: Configura threading ANTES de importar numpy/torch para evitar segfault no Mac
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Mac M1: usa 'spawn' ao invés de 'fork' para evitar segfault
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Configura multiprocessing para Mac M1
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Já foi configurado

# --- torna o repo importável mesmo se chamado de qualquer lugar
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.loader import load_beir_dataset, select_split, as_documents, as_queries
from src.eval.evaluator import evaluate_predictions
from src.vectorizers.tri_modal_vectorizer import TriModalVectorizer
from src.indexes.hybrid_index import HybridIndex
from src.utils.io import ensure_dir
from src.utils.logging import get_logger


def _default_dataset_root(name: str) -> Path:
    return ROOT / "data" / name / "processed" / "beir"


def _ner_defaults_for(dataset_name: str):
    """Heurística leve de NER/labels por dataset."""
    if dataset_name.lower() == "scifact":
        return dict(
            ner_backend="scispacy",
            ner_model=None,
            ner_allowed_labels=["DISEASE", "CHEMICAL"],  # se BC5CDR estiver instalado
        )
    return dict(ner_backend="spacy", ner_model=None, ner_allowed_labels=None)


def dataset_stats_both(dataset_root: Path) -> Dict[str, int]:
    """Retorna estatísticas em dois formatos: 'paper-like' e 'BEIR-like'."""
    corpus, queries, qrels = load_beir_dataset(dataset_root)
    split = select_split(qrels, ("test", "dev", "validation", "train"))
    split_eval = "test" if "test" in set(qrels["split"]) else split
    qrels_eval = qrels[qrels["split"] == split_eval].copy()

    # paper-like: Qrels* := número de QUERIES no split avaliado
    test_queries = qrels_eval["query_id"].nunique()

    # BEIR-like: pares (query, doc) relevantes no split avaliado
    qrels_pairs = len(qrels_eval)

    return {
        "corpus_size": len(corpus),
        "queries_total": len(queries),
        "qrels_star_paper_like": test_queries,   # o que o paper chama de "Qrels"
        "qrels_test_pairs_beir": qrels_pairs,    # BEIR (pares)
        "split_eval": split_eval,
    }


def _run_once(
    dataset_root: Path,
    semantic_model: str,
    graph_model: str,
    tfidf_dim: int,
    tfidf_backend: str,
    device: Optional[str],
    entity_artifacts: Optional[Path],
    entity_force_rebuild: bool,
    ner_batch_size: int,
    ner_n_process: int,
    index_artifacts: Optional[Path],
    faiss_factory: Optional[str],
    faiss_metric: str,
    faiss_nprobe: Optional[int],
    faiss_train_size: int,
    k_eval: int,
) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, float]]]]:
    """Executa a fase pré-rerank (índice híbrido) para um modelo semântico."""
    corpus, queries, qrels = load_beir_dataset(dataset_root)
    # split preferido: test > dev/validation > train
    split = select_split(qrels, ("test", "dev", "validation", "train"))
    split_eval = "test" if "test" in set(qrels["split"]) else split
    qrels_eval = qrels[qrels["split"] == split_eval].copy()

    docs = as_documents(corpus)
    qids = set(qrels_eval["query_id"].unique().tolist())
    queries_eval = queries[queries["query_id"].isin(qids)]
    qlist = as_queries(queries_eval)

    # defaults de NER por dataset
    ds_name = dataset_root.parent.parent.name  # "scifact"/"fiqa"/"nfcorpus"
    ner_kwargs = _ner_defaults_for(ds_name)

    # vetorizer tri-modal
    vec = TriModalVectorizer(
        semantic_backend="hf",
        semantic_model_name=semantic_model,    # A ou B
        tfidf_backend=tfidf_backend,
        tfidf_dim=tfidf_dim,
        query_prefix="", doc_prefix="",
        graph_model_name=graph_model,          # BGE-Large no slice g
        ner_backend=ner_kwargs["ner_backend"],
        ner_model=ner_kwargs["ner_model"],
        ner_allowed_labels=ner_kwargs["ner_allowed_labels"],
        ner_batch_size=ner_batch_size,         # Otimizado para Mac M1 8GB
        ner_n_process=ner_n_process,           # Recomendado: 1 para Mac M1
        entity_artifact_dir=str(entity_artifacts) if entity_artifacts else None,
        entity_force_rebuild=entity_force_rebuild,
        device=device,
    )

    # fit no corpus
    t0 = time.time()
    vec.fit_corpus((d.title or "") + " " + (d.text or "") for d in docs)
    t_fit = time.time() - t0

    # índice FAISS
    index = HybridIndex(
        vectorizer=vec,
        faiss_factory=(faiss_factory or None),
        faiss_metric=faiss_metric,
        faiss_nprobe=faiss_nprobe,
        faiss_train_size=faiss_train_size,
        artifact_dir=str(index_artifacts) if index_artifacts else None,
        index_name="hybrid.index",
    )

    t0 = time.time()
    index.build((d.doc_id, (d.title or "") + " " + (d.text or "")) for d in docs)
    t_index = time.time() - t0

    # retrieve pré-rerank
    preds: Dict[str, List[Tuple[str, float]]] = {}
    t_retrieve = 0.0
    for q in qlist:
        t1 = time.time()
        q_vec = vec.concat(vec.encode_text(q.text, is_query=True))
        topk = index.search(q_vec, topk=max(10, k_eval))
        t_retrieve += (time.time() - t1)
        preds[q.query_id] = topk[:k_eval]

    # avaliação
    metrics = evaluate_predictions(preds, qrels_eval, ks=(k_eval,))
    metrics["config"] = semantic_model.split("/")[-1]
    metrics["split"] = split_eval
    metrics["t_fit_sec"] = round(t_fit, 3)
    metrics["t_index_sec"] = round(t_index, 3)
    metrics["t_retrieve_sec"] = round(t_retrieve, 3)
    return metrics, preds


def parse_args():
    p = argparse.ArgumentParser(description="Pré-rerank A/B (MiniLM vs BGE) no híbrido s+t+g, com stats paper-like/BEIR-like.")
    # dataset
    g = p.add_argument_group("Dataset")
    g.add_argument("--dataset", type=str, choices=["scifact", "fiqa", "nfcorpus"], default="scifact",
                   help="Nome do dataset (usa caminho padrão ./data/<name>/processed/beir).")
    g.add_argument("--dataset-root", type=str, default="",
                   help="Se quiser informar o caminho exato do dataset (prioriza sobre --dataset).")
    g.add_argument("--all", action="store_true", default=False, help="Roda A/B em todos os datasets.")
    g.add_argument("--k", type=int, default=10, help="k para métricas @k")

    # modelos
    m = p.add_argument_group("Modelos")
    m.add_argument("--semantic-a", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Modelo semântico A (MiniLM por padrão).")
    m.add_argument("--semantic-b", type=str, default="BAAI/bge-large-en-v1.5",
                   help="Modelo semântico B (BGE-Large por padrão).")
    m.add_argument("--graph-model", type=str, default="BAAI/bge-large-en-v1.5",
                   help="Modelo para embutir entidades (slice g).")
    m.add_argument("--tfidf-dim", type=int, default=1000)
    m.add_argument("--tfidf-backend", type=str, choices=["sklearn", "pyserini"], default="sklearn")
    m.add_argument("--device", type=str, default=None, help="ex.: cuda:0 ou cpu")
    
    # NER/Entities (otimizado para Mac M1 8GB)
    n = p.add_argument_group("NER e Entidades")
    n.add_argument("--ner-batch-size", type=int, default=8,
                   help="Batch size para NER (default=8, otimizado para RAM limitada)")
    n.add_argument("--ner-n-process", type=int, default=1,
                   help="Processos paralelos para NER (default=1, recomendado para Mac M1)")

    # artefatos/caches
    a = p.add_argument_group("Artefatos")
    a.add_argument("--entity-artifacts", type=str, default="",
                   help="Pasta para cachear IDF/embeddings de entidade.")
    a.add_argument("--entity-force-rebuild", action="store_true", default=False,
                   help="Força rebuild do cache de embeddings de entidades (útil se dimensões mudaram).")
    a.add_argument("--index-artifacts", type=str, default="",
                   help="Pasta para salvar/carregar índice FAISS.")
    a.add_argument("--csv-out", type=str, default="", help="Salvar tabela A/B em CSV (linha por modelo/dataset).")

    # FAISS
    f = p.add_argument_group("FAISS")
    f.add_argument("--faiss-factory", type=str, default="",
                   help='Ex.: "OPQ64,IVF4096,PQ64x8" (vazio => FlatIP).')
    f.add_argument("--faiss-metric", type=str, choices=["ip", "l2"], default="ip")
    f.add_argument("--faiss-nprobe", type=int, default=0, help="0 => default do índice.")
    f.add_argument("--faiss-train-size", type=int, default=0, help="0 => auto (>= 30*nlist).")

    return p.parse_args()


def main_one_dataset(ds_name: str, ds_root: Path, args) -> pd.DataFrame:
    log = get_logger(f"ab_prerank[{ds_name}]")

    # diretórios de artefatos
    out_base = ROOT / "outputs" / "artifacts"
    ent_dir = Path(args.entity_artifacts).resolve() if args.entity_artifacts else (out_base / f"{ds_name}_entities")
    idx_dir = Path(args.index_artifacts).resolve() if args.index_artifacts else (out_base / f"{ds_name}_index_{'ivf' if args.faiss_factory else 'flat'}")
    ent_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)

    # === Estatísticas "paper-like" x "BEIR-like" ===
    stats = dataset_stats_both(ds_root)
    paper_df = pd.DataFrame([{
        "Dataset": ds_name,
        "Corpus size": stats["corpus_size"],
        "Queries": stats["queries_total"],
        "Qrels*": stats["qrels_star_paper_like"],   # paper-like (na Tabela I do paper)
    }])
    beir_df = pd.DataFrame([{
        "Dataset": ds_name,
        "Split": stats["split_eval"],
        "qrels_test_pairs (BEIR)": stats["qrels_test_pairs_beir"],
    }])

    print("\n=== TABLE I (paper-like) ===")
    print(paper_df.to_string(index=False))
    print("\n*Qrels* aqui = número de QUERIES no split de teste (como apresentado no paper).")
    print("\n--- BEIR-like (pares relevantes) ---")
    print(beir_df.to_string(index=False))

    # === A/B pré-rerank ===
    metrics_rows = []

    for sem_model in (args.semantic_a, args.semantic_b):
        metrics, _ = _run_once(
            dataset_root=ds_root,
            semantic_model=sem_model,
            graph_model=args.graph_model,
            tfidf_dim=args.tfidf_dim,
            tfidf_backend=args.tfidf_backend,
            device=args.device,
            entity_artifacts=ent_dir,
            entity_force_rebuild=args.entity_force_rebuild,
            ner_batch_size=args.ner_batch_size,
            ner_n_process=args.ner_n_process,
            index_artifacts=idx_dir,
            faiss_factory=(args.faiss_factory or None),
            faiss_metric=args.faiss_metric,
            faiss_nprobe=(None if args.faiss_nprobe <= 0 else args.faiss_nprobe),
            faiss_train_size=args.faiss_train_size,
            k_eval=args.k,
        )
        m = metrics.iloc[0].to_dict()
        m["dataset"] = ds_name
        metrics_rows.append(m)

    df = pd.DataFrame(metrics_rows)
    # Ordena por dataset -> config
    df = df[[
        "dataset", "config", "k", "nDCG", "MRR", "MAP", "Recall", "Precision",
        "split", "t_fit_sec", "t_index_sec", "t_retrieve_sec"
    ]].sort_values(["dataset", "config"]).reset_index(drop=True)

    print("\n=== Pré-rerank A/B (híbrido s+t+g) — métricas @{} ===".format(args.k))
    # Mostra em ordem nDCG (foco do paper) + demais métricas
    with pd.option_context("display.max_columns", None):
        print(df.to_string(index=False))

    return df


def main():
    args = parse_args()

    if args.dataset_root:
        ds_root = Path(args.dataset_root).resolve()
        ds_name = ds_root.parent.parent.name
        assert ds_root.exists(), f"Dataset root não encontrado: {ds_root}"
        all_names = [ds_name]
        roots = {ds_name: ds_root}
    elif args.all:
        all_names = ["scifact", "fiqa", "nfcorpus"]
        roots = {n: _default_dataset_root(n) for n in all_names}
        for n, r in roots.items():
            assert r.exists(), f"Dataset root não encontrado: {r}"
    else:
        ds_name = args.dataset
        ds_root = _default_dataset_root(ds_name)
        assert ds_root.exists(), f"Dataset root não encontrado: {ds_root}"
        all_names = [ds_name]
        roots = {ds_name: ds_root}

    # roda e agrega resultados
    all_metrics = []
    for n in all_names:
        df = main_one_dataset(n, roots[n], args)
        all_metrics.append(df)

    out = pd.concat(all_metrics, ignore_index=True)

    # resumo estilo Tabela IV (apenas nDCG@k por dataset)
    summary = out.pivot_table(index="dataset", columns="config", values="nDCG", aggfunc="first")
    print("\n=== Resumo nDCG@{} por dataset (estilo Tabela IV, pré-rerank) ===".format(args.k))
    print(summary.to_string())

    # CSV opcional
    if args.csv_out:
        out_path = Path(args.csv_out)
        ensure_dir(out_path.parent)
        out.to_csv(out_path, index=False)
        print(f"\nSalvo CSV detalhado em: {out_path}")

if __name__ == "__main__":
    main()
