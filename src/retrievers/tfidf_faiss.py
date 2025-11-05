from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from ..datasets.schema import Document, Query
from .base import AbstractRetriever
from ..vectorizers.tfidf_vectorizer import TFIDFVectorizer
from ..utils.logging import get_logger, log_time
from ..indexes.faiss_index import FaissFlatIPIndex

_log = get_logger("retriever.tfidf")

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


class TFIDFRetriever(AbstractRetriever):
    """
    Retriever esparso via TF-IDF denso (L2-normalizado) com FAISS (IP) e persistência opcional.
    """
    def __init__(self,
                 dim: int = None,
                 min_df: int = 1,
                 backend: str = "sklearn",
                 use_faiss: bool = True,
                 artifact_dir: Optional[str] = None,
                 index_name: str = "tfidf.index"):
        self.vec = TFIDFVectorizer(dim=dim, min_df=min_df, backend=backend)
        self.doc_ids: List[str] = []
        # Para TF-IDF, usamos matriz esparsa (não densa)
        self.doc_mat_sparse = None
        self.index = None
        self.faiss_helper = FaissFlatIPIndex(artifact_dir=artifact_dir, index_name=index_name)
        self.use_faiss = (use_faiss and _HAS_FAISS)

        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.index_name = index_name
        if self.artifact_dir:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            self._index_path = self.artifact_dir / index_name
            self._ids_path = self.artifact_dir / f"{index_name}.ids.json"
        else:
            self._index_path = None
            self._ids_path = None

    def _try_load(self) -> bool:
        if not self.use_faiss:
            return False
        ok = self.faiss_helper.try_load()
        if ok:
            self.index = self.faiss_helper.index
            self.doc_ids = self.faiss_helper.doc_ids
            # Verifica se a dimensão do índice carregado corresponde à dimensão atual
            if self.index is not None:
                if self.index.d != self.vec.total_dim:
                    _log.warning(f"  ⚠️  Dimensão do índice cache ({self.index.d}) != dimensão atual ({self.vec.total_dim}). Reconstruindo índice.")
                    self.index = None
                    self.doc_ids = []
                    ok = False
        return ok

    def _save(self):
        if not (self.use_faiss and self.index):
            return
        self.faiss_helper.index = self.index
        self.faiss_helper.doc_ids = self.doc_ids
        self.faiss_helper.save()

    def build_index(self, docs: List[Document]) -> None:
        _log.info(f"🚀 Building TF-IDF Index ({len(docs)} documentos)")
        self.doc_ids = [d.doc_id for d in docs]
        texts = [(d.title or "") + " " + (d.text or "") for d in docs]

        with log_time(_log, "Fit TF-IDF no corpus"):
            self.vec.fit_corpus(texts)

        with log_time(_log, "Building sparse TF-IDF matrix"):
            # Usa o vectorizer interno do encoder para manter sparse
            self.doc_mat_sparse = self.vec.encoder._vectorizer.transform(texts)
        
        _log.info(f"  ✓ TF-IDF sparse matrix: {self.doc_mat_sparse.shape}, sparsity={1 - self.doc_mat_sparse.nnz / (self.doc_mat_sparse.shape[0] * self.doc_mat_sparse.shape[1]):.2%}")
        
        # Não usa FAISS para TF-IDF
        self.index = None
        self.use_faiss = False

    def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        from sklearn.metrics.pairwise import cosine_similarity
        
        results: Dict[str, List[Tuple[str, float]]] = {}
        if not hasattr(self, 'doc_mat_sparse') or self.doc_mat_sparse is None:
            return {q.query_id: [] for q in queries}

        for q in queries:
            # Transforma query para sparse usando o vectorizer interno
            query_sparse = self.vec.encoder._vectorizer.transform([q.text])
            
            # Calcula similaridade de cosseno (sparse-friendly)
            similarities = cosine_similarity(query_sparse, self.doc_mat_sparse).flatten()
            
            # Top-k usando argpartition (eficiente)
            top_idx = np.argpartition(similarities, -k)[-k:]
            top_idx = top_idx[np.argsort(similarities[top_idx])[::-1]]
            
            results[q.query_id] = [(self.doc_ids[i], float(similarities[i])) for i in top_idx]
        
        return results