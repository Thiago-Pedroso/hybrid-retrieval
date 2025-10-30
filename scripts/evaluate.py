from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.datasets.loader import load_beir_dataset, select_split
from src.utils.io import read_jsonl, jsonl_to_predictions, ensure_dir
from src.eval.evaluator import evaluate_predictions
from src.utils.logging import get_logger

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--predictions", type=str, required=True)
    p.add_argument("--split", type=str, default=None, help="Se None, usa preferências do arquivo qrels")
    p.add_argument("--qrels-path", type=str, default=None, help="Opcional: caminho para qrels custom (jsonl ou parquet) com coluna split")
    p.add_argument("--ks", type=str, default="1,3,5,10")
    p.add_argument("--out", type=str, default="./outputs/metrics.csv")
    return p.parse_args()

def main():
    args = parse_args()
    log = get_logger()
    root = Path(args.dataset_root)
    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]

    # Carrega dataset
    corpus, queries, qrels = load_beir_dataset(root)
    # Qrels custom opcional
    if args.qrels_path:
        qp = Path(args.qrels_path)
        if qp.suffix.lower() == ".jsonl":
            qrels = pd.read_json(qp, lines=True)
        elif qp.suffix.lower() == ".parquet":
            qrels = pd.read_parquet(qp, engine="pyarrow")
        else:
            raise ValueError(f"Formato não suportado para qrels: {qp}")
        # Normalizações mínimas
        qrels["doc_id"] = qrels["doc_id"].astype(str)
        qrels["query_id"] = qrels["query_id"].astype(str)
    split = args.split or select_split(qrels)
    qrels_split = qrels[qrels["split"] == split].copy()
    log.info(f"Avaliando contra split={split} | qrels linhas={len(qrels_split)}")

    rows = read_jsonl(Path(args.predictions))
    preds = jsonl_to_predictions(rows)

    agg = evaluate_predictions(preds, qrels_split, ks=tuple(ks))
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    agg.to_csv(out_path, index=False)
    log.info(f"Métricas salvas em: {out_path}\n{agg}")

if __name__ == "__main__":
    main()
