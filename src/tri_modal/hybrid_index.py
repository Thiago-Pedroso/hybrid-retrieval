from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Iterable

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from .vectorizer import TriModalVectorizer

class HybridIndex:
    def __init__(self, vectorizer: TriModalVectorizer):
        self.vec = vectorizer
        self.doc_ids: List[str] = []
        self.doc_mat: np.ndarray = np.zeros((0, 1), dtype=np.float32)
        self.index = None

    def build(self, doc_id_and_text: Iterable[Tuple[str, str]]):
        ids = []
        vecs = []
        for doc_id, text in doc_id_and_text:
            parts = self.vec.encode_text(text)
            v = self.vec.concat(parts)
            ids.append(doc_id)
            vecs.append(v)
        self.doc_ids = ids
        self.doc_mat = np.vstack(vecs).astype(np.float32)

        if _HAS_FAISS:
            d = self.doc_mat.shape[1]
            self.index = faiss.IndexFlatIP(d)  # cos ~ dot (com vetores L2-normalizados)
            self.index.add(self.doc_mat)
        else:
            self.index = None  # usaremos NumPy

    def search(self, query_vec: np.ndarray, topk: int = 150) -> List[Tuple[str, float]]:
        q = query_vec.reshape(1, -1).astype(np.float32)
        if _HAS_FAISS and self.index is not None:
            scores, idx = self.index.search(q, topk)
            idx = idx[0].tolist()
            scores = scores[0].tolist()
        else:
            # NumPy IP
            sims = (self.doc_mat @ q.T).reshape(-1)
            idx = np.argpartition(sims, -topk)[-topk:]
            idx = idx[np.argsort(sims[idx])[::-1]]
            scores = sims[idx].tolist()

        return [(self.doc_ids[i], float(scores[j])) for j, i in enumerate(idx)]
