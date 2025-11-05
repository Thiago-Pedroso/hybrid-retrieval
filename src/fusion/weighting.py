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