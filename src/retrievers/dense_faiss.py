from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from ..datasets.schema import Document, Query
from .base import AbstractRetriever
from ..vectorizers.dense_vectorizer import DenseVectorizer
from ..indexes.faiss_index import FaissFlatIPIndex
from ..utils.logging import get_logger, log_time

_log = get_logger("retriever.dense")

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


class DenseFaiss(AbstractRetriever):
    """
    Retriever denso oficial (MiniLM ou BGE):
    - Embeddings via Hugging Face (SentenceTransformers/Transformers)
    - Normalização L2 e similaridade por inner-product (≈ cosseno)
    - Índice FAISS (IndexFlatIP) com fallback NumPy
    - Persistência opcional do índice e mapeamento de IDs
    """
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 query_prefix: str = "",
                 doc_prefix: str = "",
                 # FAISS
                 use_faiss: bool = True,
                 artifact_dir: Optional[str] = None,
                 index_name: str = "dense.index"):
        self.vec = DenseVectorizer(
            model_name=model_name,
            device=device,
            query_prefix=query_prefix,
            doc_prefix=doc_prefix,
        )
        self.dim = int(self.vec.total_dim)
        self.doc_ids: List[str] = []
        self.doc_mat: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self.index = None
        self.faiss_helper = FaissFlatIPIndex(artifact_dir=artifact_dir, index_name=index_name)
        self.use_faiss = (use_faiss and _HAS_FAISS)

        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.index_name = index_name

    # ------------------------ persistência ------------------------

    def _try_load(self) -> bool:
        if not (self.use_faiss):
            return False
        ok = self.faiss_helper.try_load()
        if ok:
            self.index = self.faiss_helper.index
            self.doc_ids = self.faiss_helper.doc_ids
        return ok

    def _save(self):
        if not (self.use_faiss and self.index):
            return
        self.faiss_helper.index = self.index
        self.faiss_helper.doc_ids = self.doc_ids
        self.faiss_helper.save()

    # ------------------------ build ------------------------

    def build_index(self, docs: List[Document]) -> None:
        _log.info(f"🚀 Building Dense Index ({len(docs)} documentos)")
        self.doc_ids = [d.doc_id for d in docs]
        texts = [(d.title or "") + " " + (d.text or "") for d in docs]

        with log_time(_log, "Encoding documents"):
            vecs = [self.vec.encode_doc(t) for t in texts]
            self.doc_mat = np.stack(vecs, axis=0).astype(np.float32) if vecs else np.zeros((0, self.dim), dtype=np.float32)

        # tenta carregar índice salvo (se existir)
        if self._try_load():
            _log.info(f"  ✓ Índice carregado do cache: {self._index_path}")
            return

        if self.use_faiss:
            with log_time(_log, "Construindo FAISS IndexFlatIP"):
                self.faiss_helper.build_from_matrix(self.doc_ids, self.doc_mat)
                self.index = self.faiss_helper.index
            _log.info(f"  ✓ FAISS IndexFlatIP: {self.index.ntotal} vetores, dim={self.dim}")
            self._save()
        else:
            self.index = None
            _log.warning("  ⚠️  FAISS indisponível, usando NumPy fallback")

    # ------------------------ search ------------------------

    def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        results: Dict[str, List[Tuple[str, float]]] = {}
        if len(self.doc_mat) == 0:
            return {q.query_id: [] for q in queries}

        for q in queries:
            qv = self.vec.encode_query(q.text).reshape(1, -1).astype(np.float32)
            if self.use_faiss and self.index is not None:
                scores, idx = self.index.search(qv, k)
                ids = [self.doc_ids[i] for i in idx[0].tolist()]
                sc = scores[0].tolist()
                results[q.query_id] = list(zip(ids, sc))
            else:
                sims = (self.doc_mat @ qv.T).reshape(-1)
                order = np.argpartition(sims, -k)[-k:]
                order = order[np.argsort(sims[order])[::-1]]
                results[q.query_id] = [(self.doc_ids[i], float(sims[i])) for i in order]
        return results
