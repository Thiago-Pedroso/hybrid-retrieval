"""
Evaluation module for retrieval predictions.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Sequence
import pandas as pd
from .metrics import (
    get_metric,
    METRICS_REGISTRY,
    # Backward compatibility imports
    mrr_at_k,
    ndcg_at_k,
    average_precision_at_k,
    recall_at_k,
    precision_at_k,
)
from ..core.interfaces import AbstractMetric
from ..datasets.schema import Query


def evaluate_predictions(
    preds: Dict[str, List[Tuple[str, float]]],
    qrels: pd.DataFrame,
    ks: Sequence[int] = (1, 3, 5, 10),
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Evaluate predictions against qrels.
    
    Args:
        preds: Dictionary mapping query_id to list of (doc_id, score) tuples
        qrels: DataFrame with columns query_id, doc_id, score, split
        ks: Sequence of k values to evaluate at
        metrics: Optional list of metric names. If None, uses all available metrics.
        
    Returns:
        DataFrame with aggregated metrics by k
    """
    # Build qrels map by query
    qrels_map = {}
    for row in qrels.itertuples(index=False):
        qid = str(getattr(row, "query_id"))
        did = str(getattr(row, "doc_id"))
        sc = float(getattr(row, "score", 1))
        qrels_map.setdefault(qid, {})[did] = sc
    
    # Get metrics to compute
    if metrics is None:
        metrics_to_compute = list(METRICS_REGISTRY.keys())
    else:
        # Normalize metric names (case-insensitive, but preserve "nDCG" format)
        metrics_to_compute = []
        for m in metrics:
            m_upper = m.upper()
            # Map common variations to registry keys
            metric_map = {
                "NDCG": "nDCG",
                "MRR": "MRR",
                "MAP": "MAP",
                "RECALL": "Recall",
                "PRECISION": "Precision",
            }
            metric_name = metric_map.get(m_upper, m_upper)
            if metric_name not in METRICS_REGISTRY:
                # Try exact match
                if m in METRICS_REGISTRY:
                    metric_name = m
                else:
                    raise ValueError(
                        f"Invalid metric: {m}. Available: {list(METRICS_REGISTRY.keys())}"
                    )
            metrics_to_compute.append(metric_name)
    
    metric_instances = {name: get_metric(name) for name in metrics_to_compute}
    
    # Compute metrics per query and k
    rows = []
    for qid, ranked in preds.items():
        ranked_ids = [d for d, _ in ranked]
        gains = qrels_map.get(qid, {})
        
        for k in ks:
            row = {"query_id": qid, "k": k}
            for metric_name, metric_instance in metric_instances.items():
                row[metric_name] = metric_instance.compute(ranked_ids, gains, k=k)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Aggregate by k (mean across queries)
    if rows:
        agg_cols = {m: "mean" for m in metrics_to_compute}
        agg = df.groupby("k").agg(agg_cols).reset_index()
        agg = agg.sort_values("k")
    else:
        # Empty results
        agg = pd.DataFrame({"k": list(ks)})
        for m in metrics_to_compute:
            agg[m] = 0.0
    
    return agg
