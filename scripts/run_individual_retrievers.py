"""
Benchmark individual de retrievers (dense, tfidf, graph) sem construir índice trimodal.

Cada retriever é avaliado independentemente com seu próprio índice FAISS:
- Dense: embeddings semânticos apenas (slice 's')
- TF-IDF: vetores TF-IDF apenas (slice 't')
- Graph: embeddings de entidades apenas (slice 'g')

Mantém estatísticas "paper-like" e "BEIR-like" compatíveis com run_prerank_ab.py.

Exemplos:
  python scripts/run_individual_retrievers.py --dataset scifact --retrievers dense,tfidf,graph --k 10
  python scripts/run_individual_retrievers.py --dataset scifact --retrievers dense --semantic-model "BAAI/bge-large-en-v1.5"
  python scripts/run_individual_retrievers.py --all --retrievers dense,tfidf,graph --k 10 --csv-out ./outputs/individual_all.csv
"""

from __future__ import annotations

import os, platform
if platform.system() == 'Darwin':
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # Evita problemas com MPS no macOS ao carregar modelos
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    # Limita threads do PyTorch no macOS
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # Desativa tokenizers paralelism do Transformers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# No macOS, força CPU por padrão para evitar segmentation faults com MPS
if platform.system() == 'Darwin':
    _DEFAULT_DEVICE = "cpu"
else:
    _DEFAULT_DEVICE = None

# --- torna o repo importável mesmo se chamado de qualquer lugar
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.loader import load_beir_dataset, select_split, as_documents, as_queries
from src.eval.evaluator import evaluate_predictions
from src.retrievers.dense_faiss import DenseFaiss
from src.retrievers.tfidf_faiss import TFIDFRetriever
from src.retrievers.graph_faiss import GraphRetriever
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


def _run_dense(
    dataset_root: Path,
    qrels_path: Optional[Path],
    semantic_model: str,
    device: Optional[str],
    index_artifacts: Optional[Path],
    k_eval: int,
) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, float]]]]:
    """Executa benchmark para retriever denso."""
    corpus, queries, qrels = load_beir_dataset(dataset_root)
    # Qrels custom opcional
    if qrels_path:
        qp = Path(qrels_path)
        if qp.suffix.lower() == ".jsonl":
            qrels = pd.read_json(qp, lines=True)
        elif qp.suffix.lower() == ".parquet":
            qrels = pd.read_parquet(qp, engine="pyarrow")
        else:
            raise ValueError(f"Formato não suportado para qrels: {qp}")
        qrels["doc_id"] = qrels["doc_id"].astype(str)
        qrels["query_id"] = qrels["query_id"].astype(str)
    # split preferido: test > dev/validation > train
    split = select_split(qrels, ("test", "dev", "validation", "train"))
    split_eval = "test" if "test" in set(qrels["split"]) else split
    qrels_eval = qrels[qrels["split"] == split_eval].copy()

    docs = as_documents(corpus)
    qids = set(qrels_eval["query_id"].unique().tolist())
    queries_eval = queries[queries["query_id"].isin(qids)]
    qlist = as_queries(queries_eval)

    # retriever denso - usa device padrão (CPU no macOS) se não especificado
    actual_device = device if device is not None else _DEFAULT_DEVICE
    log = get_logger("individual.dense")
    if platform.system() == 'Darwin' and actual_device == "cpu" and device is None:
        log.info("macOS detectado: usando device='cpu' por padrão para evitar problemas com MPS")
    log.info(f"Inicializando DenseFaiss com model={semantic_model}, device={actual_device}")
    
    retr = DenseFaiss(
        model_name=semantic_model,
        device=actual_device,
        query_prefix="",
        doc_prefix="",
        artifact_dir=str(index_artifacts / "dense") if index_artifacts else None,
        index_name="dense.index",
    )

    # fit + build
    t0 = time.time()
    retr.build_index(docs)
    t_build = time.time() - t0

    # retrieve
    t0 = time.time()
    preds = retr.retrieve(qlist, k=k_eval)
    t_retrieve = time.time() - t0

    # avaliação
    metrics = evaluate_predictions(preds, qrels_eval, ks=(k_eval,))
    metrics["config"] = f"dense-{semantic_model.split('/')[-1]}"
    metrics["retriever"] = "dense"
    metrics["split"] = split_eval
    metrics["t_build_sec"] = round(t_build, 3)
    metrics["t_retrieve_sec"] = round(t_retrieve, 3)
    return metrics, preds


def _run_tfidf(
    dataset_root: Path,
    qrels_path: Optional[Path],
    tfidf_dim: int,
    tfidf_backend: str,
    index_artifacts: Optional[Path],
    k_eval: int,
) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, float]]]]:
    """Executa benchmark para retriever TF-IDF."""
    corpus, queries, qrels = load_beir_dataset(dataset_root)
    # Qrels custom opcional
    if qrels_path:
        qp = Path(qrels_path)
        if qp.suffix.lower() == ".jsonl":
            qrels = pd.read_json(qp, lines=True)
        elif qp.suffix.lower() == ".parquet":
            qrels = pd.read_parquet(qp, engine="pyarrow")
        else:
            raise ValueError(f"Formato não suportado para qrels: {qp}")
        qrels["doc_id"] = qrels["doc_id"].astype(str)
        qrels["query_id"] = qrels["query_id"].astype(str)
    # split preferido: test > dev/validation > train
    split = select_split(qrels, ("test", "dev", "validation", "train"))
    split_eval = "test" if "test" in set(qrels["split"]) else split
    qrels_eval = qrels[qrels["split"] == split_eval].copy()

    docs = as_documents(corpus)
    qids = set(qrels_eval["query_id"].unique().tolist())
    queries_eval = queries[queries["query_id"].isin(qids)]
    qlist = as_queries(queries_eval)

    # retriever TF-IDF
    retr = TFIDFRetriever(
        dim=None,
        min_df=1,
        backend=tfidf_backend,
        artifact_dir=str(index_artifacts / "tfidf") if index_artifacts else None,
        index_name="tfidf.index",
    )

    # fit + build
    t0 = time.time()
    retr.build_index(docs)
    t_build = time.time() - t0

    # retrieve
    t0 = time.time()
    preds = retr.retrieve(qlist, k=k_eval)
    t_retrieve = time.time() - t0

    # avaliação
    metrics = evaluate_predictions(preds, qrels_eval, ks=(k_eval,))
    metrics["config"] = f"tfidf-{tfidf_backend}-{tfidf_dim}"
    metrics["retriever"] = "tfidf"
    metrics["split"] = split_eval
    metrics["t_build_sec"] = round(t_build, 3)
    metrics["t_retrieve_sec"] = round(t_retrieve, 3)
    return metrics, preds


def _run_graph(
    dataset_root: Path,
    qrels_path: Optional[Path],
    graph_model: str,
    device: Optional[str],
    entity_artifacts: Optional[Path],
    entity_force_rebuild: bool,
    ner_batch_size: int,
    ner_n_process: int,
    ds_name: str,
    index_artifacts: Optional[Path],
    k_eval: int,
) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, float]]]]:
    """Executa benchmark para retriever baseado em entidades (graph)."""
    corpus, queries, qrels = load_beir_dataset(dataset_root)
    # Qrels custom opcional
    if qrels_path:
        qp = Path(qrels_path)
        if qp.suffix.lower() == ".jsonl":
            qrels = pd.read_json(qp, lines=True)
        elif qp.suffix.lower() == ".parquet":
            qrels = pd.read_parquet(qp, engine="pyarrow")
        else:
            raise ValueError(f"Formato não suportado para qrels: {qp}")
        qrels["doc_id"] = qrels["doc_id"].astype(str)
        qrels["query_id"] = qrels["query_id"].astype(str)
    # split preferido: test > dev/validation > train
    split = select_split(qrels, ("test", "dev", "validation", "train"))
    split_eval = "test" if "test" in set(qrels["split"]) else split
    qrels_eval = qrels[qrels["split"] == split_eval].copy()

    docs = as_documents(corpus)
    qids = set(qrels_eval["query_id"].unique().tolist())
    queries_eval = queries[queries["query_id"].isin(qids)]
    qlist = as_queries(queries_eval)

    # defaults de NER por dataset
    ner_kwargs = _ner_defaults_for(ds_name)

    # retriever graph - usa device padrão (CPU no macOS) se não especificado
    actual_device = device if device is not None else _DEFAULT_DEVICE
    retr = GraphRetriever(
        graph_model_name=graph_model,
        device=actual_device,
        ner_backend=ner_kwargs["ner_backend"],
        ner_model=ner_kwargs["ner_model"],
        ner_use_noun_chunks=True,
        ner_batch_size=ner_batch_size,
        ner_n_process=ner_n_process,
        ner_allowed_labels=ner_kwargs["ner_allowed_labels"],
        min_df=2,
        entity_artifact_dir=str(entity_artifacts) if entity_artifacts else None,
        entity_force_rebuild=entity_force_rebuild,
        artifact_dir=str(index_artifacts / "graph") if index_artifacts else None,
        index_name="graph.index",
    )

    # fit + build
    t0 = time.time()
    retr.build_index(docs)
    t_build = time.time() - t0

    # retrieve
    t0 = time.time()
    preds = retr.retrieve(qlist, k=k_eval)
    t_retrieve = time.time() - t0

    # avaliação
    metrics = evaluate_predictions(preds, qrels_eval, ks=(k_eval,))
    metrics["config"] = f"graph-{graph_model.split('/')[-1]}"
    metrics["retriever"] = "graph"
    metrics["split"] = split_eval
    metrics["t_build_sec"] = round(t_build, 3)
    metrics["t_retrieve_sec"] = round(t_retrieve, 3)
    return metrics, preds


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark individual de retrievers (dense/tfidf/graph) sem índice trimodal.")
    # dataset
    g = p.add_argument_group("Dataset")
    g.add_argument("--dataset", type=str, choices=["scifact", "fiqa", "nfcorpus"], default="scifact",
                   help="Nome do dataset (usa caminho padrão ./data/<name>/processed/beir).")
    g.add_argument("--dataset-root", type=str, default="",
                   help="Se quiser informar o caminho exato do dataset (prioriza sobre --dataset).")
    g.add_argument("--all", action="store_true", default=False, help="Roda benchmark em todos os datasets.")
    g.add_argument("--k", type=int, default=10, help="k para métricas @k")
    g.add_argument("--qrels-path", type=str, default="",
                   help="Opcional: caminho para qrels custom (jsonl/parquet) com coluna 'split'.")
    
    # retrievers
    r = p.add_argument_group("Retrievers")
    r.add_argument("--retrievers", type=str, default="dense,tfidf,graph",
                   help="Lista de retrievers a rodar (ex.: 'dense,tfidf,graph' ou 'dense').")

    # modelos
    m = p.add_argument_group("Modelos")
    m.add_argument("--semantic-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Modelo semântico para retriever denso.")
    m.add_argument("--graph-model", type=str, default="BAAI/bge-large-en-v1.5",
                   help="Modelo para embutir entidades (retriever graph).")
    m.add_argument("--tfidf-dim", type=int, default=1000)
    m.add_argument("--tfidf-backend", type=str, choices=["sklearn", "pyserini"], default="sklearn")
    m.add_argument("--device", type=str, default=None, help="ex.: cuda:0 ou cpu")
    
    n = p.add_argument_group("NER e Entidades")
    n.add_argument("--ner-batch-size", type=int, default=32,
                   help="Batch size para NER")
    n.add_argument("--ner-n-process", type=int, default=4,
                   help="Processos paralelos para NER")

    # artefatos/caches
    a = p.add_argument_group("Artefatos")
    a.add_argument("--entity-artifacts", type=str, default="",
                   help="Pasta para cachear IDF/embeddings de entidade (usado pelo graph).")
    a.add_argument("--entity-force-rebuild", action="store_true", default=False,
                   help="Força rebuild do cache de embeddings de entidades.")
    a.add_argument("--index-artifacts", type=str, default="",
                   help="Pasta base para salvar/carregar índices FAISS (subpastas por retriever).")
    a.add_argument("--csv-out", type=str, default="", help="Salvar tabela de resultados em CSV.")

    return p.parse_args()


def main_one_dataset(ds_name: str, ds_root: Path, args) -> pd.DataFrame:
    log = get_logger(f"individual[{ds_name}]")

    # parse retrievers
    retrievers_to_run = [r.strip().lower() for r in args.retrievers.split(",") if r.strip()]
    valid_retrievers = {"dense", "tfidf", "graph"}
    for r in retrievers_to_run:
        if r not in valid_retrievers:
            raise ValueError(f"Retriever inválido: {r}. Deve ser um de: {valid_retrievers}")
    
    # diretórios de artefatos
    out_base = ROOT / "outputs" / "artifacts"
    ent_dir = Path(args.entity_artifacts).resolve() if args.entity_artifacts else (out_base / f"{ds_name}_entities")
    idx_dir_base = Path(args.index_artifacts).resolve() if args.index_artifacts else (out_base / f"{ds_name}_individual_indexes")
    ent_dir.mkdir(parents=True, exist_ok=True)
    idx_dir_base.mkdir(parents=True, exist_ok=True)

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

    # === Benchmarks individuais ===
    metrics_rows = []
    qrels_path = Path(args.qrels_path).resolve() if args.qrels_path else None

    if "dense" in retrievers_to_run:
        log.info("🟦 Rodando benchmark DENSE...")
        metrics, _ = _run_dense(
            dataset_root=ds_root,
            qrels_path=qrels_path,
            semantic_model=args.semantic_model,
            device=args.device,
            index_artifacts=idx_dir_base,
            k_eval=args.k,
        )
        m = metrics.iloc[0].to_dict()
        m["dataset"] = ds_name
        metrics_rows.append(m)

    if "tfidf" in retrievers_to_run:
        log.info("🟨 Rodando benchmark TF-IDF...")
        metrics, _ = _run_tfidf(
            dataset_root=ds_root,
            qrels_path=qrels_path,
            tfidf_dim=args.tfidf_dim,
            tfidf_backend=args.tfidf_backend,
            index_artifacts=idx_dir_base,
            k_eval=args.k,
        )
        m = metrics.iloc[0].to_dict()
        m["dataset"] = ds_name
        metrics_rows.append(m)

    if "graph" in retrievers_to_run:
        log.info("🟩 Rodando benchmark GRAPH...")
        metrics, _ = _run_graph(
            dataset_root=ds_root,
            qrels_path=qrels_path,
            graph_model=args.graph_model,
            device=args.device,
            entity_artifacts=ent_dir,
            entity_force_rebuild=args.entity_force_rebuild,
            ner_batch_size=args.ner_batch_size,
            ner_n_process=args.ner_n_process,
            ds_name=ds_name,
            index_artifacts=idx_dir_base,
            k_eval=args.k,
        )
        m = metrics.iloc[0].to_dict()
        m["dataset"] = ds_name
        metrics_rows.append(m)

    if not metrics_rows:
        raise ValueError("Nenhum retriever selecionado para rodar!")

    df = pd.DataFrame(metrics_rows)
    # Ordena por dataset -> retriever -> config
    cols_to_show = ["dataset", "retriever", "config", "k", "nDCG", "MRR", "MAP", "Recall", "Precision",
                    "split", "t_build_sec", "t_retrieve_sec"]
    available_cols = [c for c in cols_to_show if c in df.columns]
    df = df[available_cols].sort_values(["dataset", "retriever", "config"]).reset_index(drop=True)

    print(f"\n=== Benchmarks Individuais — métricas @{args.k} ===")
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

    # resumo por retriever
    if "nDCG" in out.columns:
        summary = out.pivot_table(index="dataset", columns="retriever", values="nDCG", aggfunc="first")
        print(f"\n=== Resumo nDCG@{args.k} por dataset e retriever ===")
        print(summary.to_string())

    # CSV opcional
    if args.csv_out:
        out_path = Path(args.csv_out)
        ensure_dir(out_path.parent)
        out.to_csv(out_path, index=False)
        print(f"\nSalvo CSV detalhado em: {out_path}")

if __name__ == "__main__":
    main()

