"""
Evaluation metrics for information retrieval.

This module provides class-based implementations implementing AbstractMetric interface.
"""

import numpy as np
from typing import Dict, List, Sequence
from ..core.interfaces import AbstractMetric


# ============================================================================
# Helper functions (used by both classes and standalone functions)
# ============================================================================

def _dcg_at_k(ranked: Sequence[str], gains: Dict[str, float], k: int = 10) -> float:
    """Compute DCG@k."""
    dcg = 0.0
    for i, did in enumerate(ranked[:k], start=1):
        g = gains.get(did, 0.0)
        if g > 0:
            dcg += (2**g - 1) / np.log2(i + 1)
    return dcg


# ============================================================================
# Metric Classes (implementing AbstractMetric)
# ============================================================================

class MRRMetric(AbstractMetric):
    """Mean Reciprocal Rank metric."""
    
    @property
    def name(self) -> str:
        return "MRR"
    
    def compute(
        self,
        ranked: List[str],
        gold: Dict[str, float],
        k: int = 10,
    ) -> float:
        """Compute MRR@k."""
        gold_set = {d for d, s in gold.items() if s > 0}
        for i, did in enumerate(ranked[:k], start=1):
            if did in gold_set:
                return 1.0 / i
        return 0.0


class NDCGMetric(AbstractMetric):
    """Normalized Discounted Cumulative Gain metric."""
    
    @property
    def name(self) -> str:
        return "nDCG"
    
    def compute(
        self,
        ranked: List[str],
        gold: Dict[str, float],
        k: int = 10,
    ) -> float:
        """Compute nDCG@k."""
        ideal = sorted([s for s in gold.values() if s > 0], reverse=True)[:k]
        idcg = 0.0
        for i, g in enumerate(ideal, start=1):
            idcg += (2**g - 1) / np.log2(i + 1)
        
        if idcg == 0:
            return 0.0
        
        dcg = _dcg_at_k(ranked, gold, k)
        return dcg / idcg


class MAPMetric(AbstractMetric):
    """Mean Average Precision metric."""
    
    @property
    def name(self) -> str:
        return "MAP"
    
    def compute(
        self,
        ranked: List[str],
        gold: Dict[str, float],
        k: int = 10,
    ) -> float:
        """Compute MAP@k."""
        gold_set = {d for d, s in gold.items() if s > 0}
        if not gold_set:
            return 0.0
        
        hits = 0
        precision_sum = 0.0
        for i, did in enumerate(ranked[:k], start=1):
            if did in gold_set:
                hits += 1
                precision_sum += hits / i
        
        return precision_sum / min(len(gold_set), k)


class RecallMetric(AbstractMetric):
    """Recall metric."""
    
    @property
    def name(self) -> str:
        return "Recall"
    
    def compute(
        self,
        ranked: List[str],
        gold: Dict[str, float],
        k: int = 10,
    ) -> float:
        """Compute Recall@k."""
        gold_set = {d for d, s in gold.items() if s > 0}
        if not gold_set:
            return 0.0
        return len(set(ranked[:k]) & gold_set) / len(gold_set)


class PrecisionMetric(AbstractMetric):
    """Precision metric."""
    
    @property
    def name(self) -> str:
        return "Precision"
    
    def compute(
        self,
        ranked: List[str],
        gold: Dict[str, float],
        k: int = 10,
    ) -> float:
        """Compute Precision@k."""
        if k == 0:
            return 0.0
        gold_set = {d for d, s in gold.items() if s > 0}
        return len(set(ranked[:k]) & gold_set) / k


# ============================================================================
# Registry of available metrics
# ============================================================================

METRICS_REGISTRY: Dict[str, AbstractMetric] = {
    "MRR": MRRMetric(),
    "nDCG": NDCGMetric(),
    "MAP": MAPMetric(),
    "Recall": RecallMetric(),
    "Precision": PrecisionMetric(),
}


def get_metric(name: str) -> AbstractMetric:
    """Get metric by name (case-insensitive, but preserves exact matches)."""
    # Try exact match first (important for "nDCG")
    if name in METRICS_REGISTRY:
        return METRICS_REGISTRY[name]
    
    # Try case-insensitive match with mapping
    name_upper = name.upper()
    metric_map = {
        "NDCG": "nDCG",
        "MRR": "MRR",
        "MAP": "MAP",
        "RECALL": "Recall",
        "PRECISION": "Precision",
    }
    normalized = metric_map.get(name_upper, name_upper)
    
    if normalized in METRICS_REGISTRY:
        return METRICS_REGISTRY[normalized]
    
    raise ValueError(f"Unknown metric: {name}. Available: {list(METRICS_REGISTRY.keys())}")


