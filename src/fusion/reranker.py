from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from ..vectorizers.tri_modal_vectorizer import TriModalVectorizer
from ..encoders.encoders import l2norm

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

class TriModalReranker:
    """
    Recalcula o score combinando cosenos por fatia (s/t/g) com pesos (ws,wt,wg).
    """
    def __init__(self, vec: TriModalVectorizer):
        self.vec = vec

    def rescore(self, query_text: str, candidate_docs: List[Tuple[str, str]], weights) -> List[Tuple[str, float]]:
        ws, wt, wg = weights
        qp = self.vec.encode_text(query_text, is_query=True)
        scores = []
        for doc_id, doc_text in candidate_docs:
            dp = self.vec.encode_text(doc_text, is_query=False)
            s = ws * cosine(qp["s"], dp["s"]) + wt * cosine(qp["t"], dp["t"]) + wg * cosine(qp["g"], dp["g"])
            scores.append((doc_id, float(s)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores