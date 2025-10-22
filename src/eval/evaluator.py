from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from .metrics import mrr_at_k, ndcg_at_k, average_precision_at_k, recall_at_k, precision_at_k
from ..datasets.schema import Query

def evaluate_predictions(preds: Dict[str, List[Tuple[str, float]]],
                         qrels: pd.DataFrame,
                         ks=(1,3,5,10)) -> pd.DataFrame:
    """Avalia um dicionário {query_id: [(doc_id, score), ...]} contra qrels."""
    # mapa por query
    qrels_map = {}
    for row in qrels.itertuples(index=False):
        qid = str(getattr(row, "query_id"))
        did = str(getattr(row, "doc_id"))
        sc  = int(getattr(row, "score", 1))
        qrels_map.setdefault(qid, {})[did] = sc

    rows = []
    for qid, ranked in preds.items():
        ranked_ids = [d for d, _ in ranked]
        gains = qrels_map.get(qid, {})
        gold = {d for d, s in gains.items() if s > 0}
        for k in ks:
            rows.append({
                "query_id": qid,
                "k": k,
                "MRR": mrr_at_k(ranked_ids, gold, k=k),
                "nDCG": ndcg_at_k(ranked_ids, gains, k=k),
                "MAP": average_precision_at_k(ranked_ids, gold, k=k),
                "Recall": recall_at_k(ranked_ids, gold, k=k),
                "Precision": precision_at_k(ranked_ids, gold, k=k),
            })
    df = pd.DataFrame(rows)
    # agregados
    agg = df.groupby("k").agg({m: "mean" for m in ["MRR","nDCG","MAP","Recall","Precision"]}).reset_index()
    agg = agg.sort_values("k")
    return agg
