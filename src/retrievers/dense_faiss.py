from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from ..datasets.schema import Document, Query
from .base import AbstractRetriever

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# Stub: encoder semântico simples determinístico (mesma ideia do TriModal)
def _hash_vec(text: str, dim=384, seed=42) -> np.ndarray:
    toks = [t for t in text.lower().split()][:128]
    acc = np.zeros(dim, dtype=np.float32)
    for t in toks:
        h = abs(hash((t, seed))) % (2**32)
        rng = np.random.default_rng(h)
        acc += rng.standard_normal(dim).astype(np.float32)
    n = np.linalg.norm(acc) + 1e-12
    return acc / n

class DenseFaissStub(AbstractRetriever):
    """Retriever denso simples: um vetor por doc/query e IP com FAISS (se disponível)."""
    def __init__(self, dim: int = 384, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self.doc_ids: List[str] = []
        self.doc_mat: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self.index = None

    def build_index(self, docs: List[Document]) -> None:
        self.doc_ids = [d.doc_id for d in docs]
        texts = [(d.title or "") + " " + (d.text or "") for d in docs]
        vecs = np.stack([_hash_vec(t, self.dim, self.seed) for t in texts], axis=0).astype(np.float32)
        self.doc_mat = vecs
        if _HAS_FAISS:
            idx = faiss.IndexFlatIP(self.dim)
            idx.add(self.doc_mat)
            self.index = idx
        else:
            self.index = None

    def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        out: Dict[str, List[Tuple[str, float]]] = {}
        for q in queries:
            qv = _hash_vec(q.text, self.dim, self.seed).reshape(1, -1).astype(np.float32)
            if _HAS_FAISS and self.index is not None:
                scores, idx = self.index.search(qv, k)
                ids = [self.doc_ids[i] for i in idx[0].tolist()]
                sc = scores[0].tolist()
                out[q.query_id] = list(zip(ids, sc))
            else:
                sims = (self.doc_mat @ qv.T).reshape(-1)
                order = np.argpartition(sims, -k)[-k:]
                order = order[np.argsort(sims[order])[::-1]]
                out[q.query_id] = [(self.doc_ids[i], float(sims[i])) for i in order]
        return out
