from __future__ import annotations
import re
from typing import Dict, Tuple
from ..core.interfaces import AbstractWeightPolicy


class StaticPolicy(AbstractWeightPolicy):
    """Static weight policy with fixed weights."""

    def __init__(self, ws: float = 0.5, wt: float = 0.3, wg: float = 0.2):
        s = max(ws, 0.0); t = max(wt, 0.0); g = max(wg, 0.0)
        z = (s + t + g) or 1.0
        self.ws, self.wt, self.wg = s/z, t/z, g/z

    def weights(self, query_text: str) -> Tuple[float, ...]:
        """Return fixed weights (semantic, tfidf, graph)."""
        return (self.ws, self.wt, self.wg)


class HeuristicLLMPolicy(AbstractWeightPolicy):
    """
    Placeholder que imita a ideia do paper: ajusta pesos por modalidade conforme a query.
    - Muitos números/termos curtos → dá mais peso ao TF-IDF (wt).
    - Muitos nomes próprios / termos longos → aumenta entidades (wg).
    - Caso geral → privilegia semântico (ws).
    """
    _num = re.compile(r"\b\d+(\.\d+)?\b")

    def weights(self, query_text: str) -> Tuple[float, ...]:
        """Return adaptive weights based on query characteristics."""
        qt = query_text or ""
        n_nums = len(self._num.findall(qt))
        long_terms = sum(1 for tok in qt.split() if len(tok) > 12)
        cap_terms  = sum(1 for tok in qt.split() if tok[:1].isupper())

        ws, wt, wg = 0.5, 0.3, 0.2
        if n_nums >= 2 or len(qt.split()) <= 4:
            wt += 0.15; ws -= 0.10
        if long_terms >= 2 or cap_terms >= 2:
            wg += 0.15; ws -= 0.10
        # normaliza
        s = max(ws, 0.05); t = max(wt, 0.05); g = max(wg, 0.05)
        z = s+t+g
        return (s/z, t/z, g/z)


class DATWeightPolicy(AbstractWeightPolicy):
    """
    Dynamic Alpha Tuning (DAT) weight policy.
    Computes adaptive alpha based on LLM judge effectiveness scores.
    
    Normalizes judge scores: e^d = e_d/5, e^b = e_b/5
    Formula: α(q) = e^d / (e^d + e^b)
    
    Edge cases:
    - If e^d = e^b = 0 → α = 0.5 (neutral)
    - If e^b = 0 and e^d > 0 → α = 1.0
    - If e^d = 0 and e^b > 0 → α = 0.0
    """
    
    def __init__(self):
        """Initialize DAT weight policy."""
        pass
    
    def compute_alpha(self, e_dense: int, e_bm25: int) -> float:
        """
        Compute alpha(q) from judge effectiveness scores.
        
        Args:
            e_dense: Dense retrieval effectiveness score (0-5)
            e_bm25: BM25 retrieval effectiveness score (0-5)
        
        Returns:
            Alpha value in [0, 1], rounded to 1 decimal place
        """
        # Normalize: divide by 5
        e_d_norm = e_dense / 5.0
        e_b_norm = e_bm25 / 5.0
        
        # Edge cases
        if e_d_norm == 0.0 and e_b_norm == 0.0:
            # Both zero: neutral
            alpha = 0.5
        elif e_b_norm == 0.0 and e_d_norm > 0.0:
            # BM25 zero, dense > 0: use only dense
            alpha = 1.0
        elif e_d_norm == 0.0 and e_b_norm > 0.0:
            # Dense zero, BM25 > 0: use only BM25
            alpha = 0.0
        else:
            # Normal case: e^d / (e^d + e^b)
            denominator = e_d_norm + e_b_norm
            if denominator == 0:
                alpha = 0.5  # Fallback
            else:
                alpha = e_d_norm / denominator
        
        # Clamp to [0, 1] and round to 1 decimal
        alpha = max(0.0, min(1.0, alpha))
        alpha = round(alpha, 1)
        
        return alpha
    
    def weights(self, query_text: str) -> Tuple[float, ...]:
        """
        This method is required by AbstractWeightPolicy interface.
        For DAT, alpha is computed per query using judge scores, not query text.
        This is a placeholder that returns (1.0, 0.0) for compatibility.
        Actual alpha computation should use compute_alpha() with judge scores.
        """
        # Return (dense_weight, bm25_weight) as tuple
        # Actual alpha is computed via compute_alpha() with judge scores
        return (1.0, 0.0)  # Placeholder