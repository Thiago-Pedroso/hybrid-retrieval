import numpy as np
from typing import Dict, List, Sequence

def mrr_at_k(ranked: Sequence[str], gold: set, k=10) -> float:
    for i, did in enumerate(ranked[:k], start=1):
        if did in gold:
            return 1.0 / i
    return 0.0

def dcg_at_k(ranked: Sequence[str], gains: Dict[str, float], k=10) -> float:
    dcg = 0.0
    for i, did in enumerate(ranked[:k], start=1):
        g = gains.get(did, 0.0)
        if g > 0:
            dcg += (2**g - 1) / np.log2(i + 1)
    return dcg

def ndcg_at_k(ranked: Sequence[str], gains: Dict[str, float], k=10) -> float:
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = 0.0
    for i, g in enumerate(ideal, start=1):
        idcg += (2**g - 1) / np.log2(i + 1)
    return 0.0 if idcg == 0 else dcg_at_k(ranked, gains, k) / idcg

def average_precision_at_k(ranked: Sequence[str], gold: set, k=10) -> float:
    hits, s = 0, 0.0
    for i, did in enumerate(ranked[:k], start=1):
        if did in gold:
            hits += 1
            s += hits / i
    return 0.0 if not gold else s / min(len(gold), k)

def recall_at_k(ranked: Sequence[str], gold: set, k=10) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)

def precision_at_k(ranked: Sequence[str], gold: set, k=10) -> float:
    if k == 0:
        return 0.0
    return len(set(ranked[:k]) & gold) / k
