from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..vectorizers.tri_modal_vectorizer import TriModalVectorizer
from ..vectorizers.bi_modal_vectorizer import BiModalVectorizer
from ..core.interfaces import AbstractReranker, AbstractVectorizer
from ..encoders.encoders import l2norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class TriModalReranker(AbstractReranker):
    """
    Reranker tri-modal: recalcula score combinando cosenos por fatia (s/t/g) com pesos (ws,wt,wg).
    """
    def __init__(self, vec: TriModalVectorizer):
        self.vec = vec

    def rescore(
        self,
        query_text: str,
        candidate_docs: List[Tuple[str, str]],
        weights: Optional[Tuple[float, ...]] = None,
    ) -> List[Tuple[str, float]]:
        """Rescore candidates using tri-modal weighted cosine."""
        if weights is None:
            weights = (0.5, 0.3, 0.2)  # Default weights
        ws, wt, wg = weights
        
        qp = self.vec.encode_text(query_text, is_query=True)
        scores = []
        for doc_id, doc_text in candidate_docs:
            dp = self.vec.encode_text(doc_text, is_query=False)
            s = ws * cosine(qp["s"], dp["s"]) + wt * cosine(qp["t"], dp["t"]) + wg * cosine(qp["g"], dp["g"])
            scores.append((doc_id, float(s)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class BiModalReranker(AbstractReranker):
    """
    Reranker bi-modal: recalcula score combinando cosenos por fatia (s/t) com pesos (ws,wt).
    """
    def __init__(self, vec: BiModalVectorizer):
        self.vec = vec

    def rescore(
        self,
        query_text: str,
        candidate_docs: List[Tuple[str, str]],
        weights: Optional[Tuple[float, ...]] = None,
    ) -> List[Tuple[str, float]]:
        """Rescore candidates using bi-modal weighted cosine."""
        if weights is None:
            weights = (0.6, 0.4)  # Default weights
        ws, wt = weights
        
        qp = self.vec.encode_text(query_text, is_query=True)
        scores = []
        for doc_id, doc_text in candidate_docs:
            dp = self.vec.encode_text(doc_text, is_query=False)
            s = ws * cosine(qp["s"], dp["s"]) + wt * cosine(qp["t"], dp["t"])
            scores.append((doc_id, float(s)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores