# src/tri_modal/hybrid_index.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Iterable, Optional
from pathlib import Path
import json
import re
import warnings

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from .vectorizer import TriModalVectorizer

def _parse_nlist(factory: str) -> int:
    m = re.search(r"IVF(\d+)", factory or "")
    return int(m.group(1)) if m else 0

def _set_nprobe(index, nprobe: int):
    # tenta setar nprobe no índice (ou no sub-índice, se for PreTransform)
    try:
        if hasattr(index, "nprobe"):
            index.nprobe = int(nprobe)
            return
        if hasattr(index, "index") and hasattr(index.index, "nprobe"):
            index.index.nprobe = int(nprobe)
            return
    except Exception:
        pass

class HybridIndex:
    def __init__(self,
                 vectorizer: TriModalVectorizer,
                 faiss_factory: Optional[str] = None,   # ex.: "OPQ64,IVF4096,PQ64x8"
                 faiss_metric: str = "ip",              # "ip" | "l2"
                 faiss_nprobe: Optional[int] = None,
                 faiss_train_size: int = 0,             # 0 => auto
                 artifact_dir: Optional[str] = None,     # onde salvar index+ids
                 index_name: str = "hybrid.index"):
        self.vec = vectorizer
        self.doc_ids: List[str] = []
        self.doc_mat: np.ndarray = np.zeros((0, 1), dtype=np.float32)
        self.index = None

        self.faiss_factory = faiss_factory
        self.faiss_metric = faiss_metric
        self.faiss_nprobe = faiss_nprobe
        self.faiss_train_size = faiss_train_size

        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.index_name = index_name
        if self.artifact_dir:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            self._index_path = self.artifact_dir / index_name
            self._ids_path = self.artifact_dir / f"{index_name}.ids.json"
        else:
            self._index_path = None
            self._ids_path = None

    # ------------------------ persistência ------------------------

    def try_load(self) -> bool:
        if not (_HAS_FAISS and self._index_path and self._index_path.exists() and self._ids_path and self._ids_path.exists()):
            return False
        try:
            self.index = faiss.read_index(str(self._index_path))
            with open(self._ids_path, "r", encoding="utf-8") as f:
                self.doc_ids = json.load(f)
            return True
        except Exception as e:
            warnings.warn(f"[HybridIndex] Falha ao carregar índice: {e}")
            return False

    def _save(self):
        if not (_HAS_FAISS and self.index and self._index_path and self._ids_path):
            return
        try:
            faiss.write_index(self.index, str(self._index_path))
            with open(self._ids_path, "w", encoding="utf-8") as f:
                json.dump(self.doc_ids, f, ensure_ascii=False)
        except Exception as e:
            warnings.warn(f"[HybridIndex] Falha ao salvar índice: {e}")

    # ------------------------ build ------------------------

    def build(self, doc_id_and_text: Iterable[Tuple[str, str]]):
        # vetoriza docs
        ids = []
        vecs = []
        for doc_id, text in doc_id_and_text:
            parts = self.vec.encode_text(text, is_query=False)
            v = self.vec.concat(parts)
            ids.append(doc_id)
            vecs.append(v)
        self.doc_ids = ids
        self.doc_mat = np.vstack(vecs).astype(np.float32)

        # tenta carregar índice salvo
        if self.try_load():
            return

        if not _HAS_FAISS or self.faiss_factory in (None, "", "FlatIP"):
            # IP com vetores L2-normalizados ≈ cosseno
            if _HAS_FAISS:
                d = self.doc_mat.shape[1]
                self.index = faiss.IndexFlatIP(d)
                self.index.add(self.doc_mat)
            else:
                self.index = None  # fallback NumPy
            return

        # IndexFactory real
        d = self.doc_mat.shape[1]
        metric = faiss.METRIC_INNER_PRODUCT if self.faiss_metric.lower() == "ip" else faiss.METRIC_L2
        self.index = faiss.index_factory(d, self.faiss_factory, metric)

        # train se necessário
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            nlist = _parse_nlist(self.faiss_factory)
            N = self.doc_mat.shape[0]
            if self.faiss_train_size and self.faiss_train_size > 0:
                train_n = min(N, self.faiss_train_size)
            else:
                base = max(30 * nlist, 10000) if nlist > 0 else min(50000, N)
                train_n = int(min(N, base))
            rng = np.random.default_rng(42)
            idx = rng.choice(N, size=train_n, replace=False)
            train_vecs = self.doc_mat[idx]
            self.index.train(train_vecs)

        self.index.add(self.doc_mat)
        if self.faiss_nprobe:
            _set_nprobe(self.index, int(self.faiss_nprobe))

        # salva índice
        self._save()

    # ------------------------ search ------------------------

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
