from __future__ import annotations
"""
Checagem de splits e simulação de 80/10/10 para datasets BEIR processados.

Uso:
  python scripts/check_dataset_splits.py --dataset scifact
  python scripts/check_dataset_splits.py --dataset-root ./data/scifact/processed/beir

Opções:
  --simulate-801010           Apenas simula e mostra contagens por split proposto (sem escrever)
  --write-split               Escreve arquivo qrels_<split>.jsonl novo com coluna 'split' 80/10/10 (cautela)
  --seed 42                   Semente para particionamento determinístico

Notas:
- Lê diretamente os arquivos do dataset (parquet/jsonl) sem injetar coluna 'split'
  para diagnosticar a situação real.
"""

import argparse
from pathlib import Path
from typing import Tuple
import pandas as pd


def _load_parquet_or_jsonl_raw(path_parquet: Path, path_jsonl: Path) -> pd.DataFrame:
    if path_parquet.exists():
        try:
            return pd.read_parquet(path_parquet, engine="pyarrow")
        except Exception as e_parq:
            print(f"[check] Aviso: falha ao ler Parquet ({e_parq}). Tentando JSONL...")
    if path_jsonl.exists():
        return pd.read_json(path_jsonl, lines=True)
    raise FileNotFoundError(f"Arquivos não encontrados: {path_parquet} | {path_jsonl}")


def load_beir_raw(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corpus = _load_parquet_or_jsonl_raw(root / "corpus.parquet", root / "corpus.jsonl")
    queries = _load_parquet_or_jsonl_raw(root / "queries.parquet", root / "queries.jsonl")
    qrels = _load_parquet_or_jsonl_raw(root / "qrels.parquet", root / "qrels.jsonl")
    # normalizações mínimas
    if "doc_id" in corpus.columns:
        corpus["doc_id"] = corpus["doc_id"].astype(str)
    if "query_id" in queries.columns:
        queries["query_id"] = queries["query_id"].astype(str)
    if "doc_id" in qrels.columns:
        qrels["doc_id"] = qrels["doc_id"].astype(str)
    if "query_id" in qrels.columns:
        qrels["query_id"] = qrels["query_id"].astype(str)
    return corpus, queries, qrels


def resolve_dataset_root(args) -> Tuple[str, Path]:
    if args.dataset_root:
        p = Path(args.dataset_root).resolve()
        assert p.exists(), f"Dataset root não encontrado: {p}"
        # tenta inferir nome pelo caminho padrão .../data/<name>/processed/beir
        try:
            name = p.parent.parent.name
        except Exception:
            name = "unknown"
        return name, p
    assert args.dataset, "Informe --dataset ou --dataset-root"
    name = args.dataset.lower()
    root = Path(__file__).resolve().parents[1] / "data" / name / "processed" / "beir"
    assert root.exists(), f"Dataset root não encontrado: {root}"
    return name, root


def analyze_splits(qrels: pd.DataFrame) -> pd.DataFrame:
    has_split = "split" in qrels.columns
    rows = []
    if has_split:
        for s, df in qrels.groupby("split"):
            rows.append({
                "split": str(s),
                "num_pairs": len(df),
                "num_queries": df["query_id"].nunique(),
            })
    else:
        rows.append({"split": "<absent>", "num_pairs": len(qrels), "num_queries": qrels["query_id"].nunique()})
    return pd.DataFrame(rows)


def simulate_801010(qrels: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    qids = sorted(qrels["query_id"].unique().tolist())
    rng = pd.Series(qids, dtype=str).sample(frac=1.0, random_state=seed).tolist()
    n = len(rng)
    n_train = int(round(0.8 * n))
    n_val = int(round(0.1 * n))
    n_test = n - n_train - n_val
    train_q = set(rng[:n_train])
    val_q = set(rng[n_train:n_train + n_val])
    test_q = set(rng[n_train + n_val:])

    def _assign(qid: str) -> str:
        if qid in train_q:
            return "train"
        if qid in val_q:
            return "validation"
        return "test"

    qrels_sim = qrels.copy()
    qrels_sim["split_sim801010"] = qrels_sim["query_id"].map(_assign)
    return qrels_sim


def report_simulation(qrels_sim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, df in qrels_sim.groupby("split_sim801010"):
        rows.append({
            "split": s,
            "num_pairs": len(df),
            "num_queries": df["query_id"].nunique(),
        })
    return pd.DataFrame(rows).sort_values("split")


def parse_args():
    p = argparse.ArgumentParser(description="Checagem de splits e simulação 80/10/10 (por query)")
    p.add_argument("--dataset", type=str, choices=["scifact", "fiqa", "nfcorpus"], default=None)
    p.add_argument("--dataset-root", type=str, default=None)
    p.add_argument("--simulate-801010", action="store_true", default=False)
    p.add_argument("--write-split", action="store_true", default=False,
                   help="Escreve qrels_<split>.jsonl com coluna 'split' 80/10/10")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    ds_name, root = resolve_dataset_root(args)
    print(f"[check] Dataset: {ds_name}")
    print(f"[check] Root:    {root}")

    corpus, queries, qrels = load_beir_raw(root)
    print(f"[check] Tamanhos: corpus={len(corpus)} | queries={len(queries)} | qrels_pairs={len(qrels)}")

    # 1) Situação atual dos splits
    print("\n=== Splits presentes (qrels) ===")
    present = analyze_splits(qrels)
    with pd.option_context("display.max_columns", None):
        print(present.to_string(index=False))

    # 2) Simulação 80/10/10
    if args.simulate_801010 or args.write_split:
        qrels_sim = simulate_801010(qrels, seed=args.seed)
        sim_tbl = report_simulation(qrels_sim)
        print("\n=== Simulação 80/10/10 (por query_id) ===")
        with pd.option_context("display.max_columns", None):
            print(sim_tbl.to_string(index=False))

        # 3) Escrita opcional
        if args.write_split:
            out_path = Path(args.out).resolve() if args.out else (root / "qrels_801010.jsonl")
            # grava JSONL com coluna 'split'
            out_df = qrels_sim.drop(columns=["split"], errors="ignore").rename(columns={"split_sim801010": "split"})
            out_df.to_json(out_path, lines=True, orient="records", force_ascii=False)
            print(f"\n[check] Escrito qrels com split 80/10/10 em: {out_path}")


if __name__ == "__main__":
    main()


