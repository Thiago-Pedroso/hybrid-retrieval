from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from ..datasets.schema import Document, Query
from .base import AbstractRetriever
from ..vectorizers.graph_vectorizer import GraphVectorizer
from ..utils.logging import get_logger, log_time
from ..indexes.faiss_index import FaissFlatIPIndex

_log = get_logger("retriever.graph")

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


class GraphRetriever(AbstractRetriever):
    """
    Retriever baseado no modo 'grafo' do paper: TF-IDF de entidades * embedding de entidade.
    - Extrai entidades via spaCy/sciSpaCy (ou fallback simples), calcula IDF no corpus
    - Embeddings de entidades via HF (ex.: BGE-Large)
    - Agrega por TF-IDF e normaliza (L2); indexa com FAISS (IP)
    """
    def __init__(self,
                 graph_model_name: str = "BAAI/bge-large-en-v1.5",
                 device: Optional[str] = None,
                 ner_backend: str = "scispacy",
                 ner_model: Optional[str] = None,
                 ner_use_noun_chunks: bool = True,
                 ner_batch_size: int = 128,
                 ner_n_process: int = 4,
                 ner_allowed_labels: Optional[List[str]] = None,
                 min_df: int = 2,
                 # FAISS
                 use_faiss: bool = True,
                 artifact_dir: Optional[str] = None,
                 index_name: str = "graph.index",
                 # Cache de entidades
                 entity_artifact_dir: Optional[str] = None,
                 entity_force_rebuild: bool = False):
        self.vec = GraphVectorizer(
            graph_model_name=graph_model_name,
            device=device,
            ner_backend=ner_backend,
            ner_model=ner_model,
            ner_use_noun_chunks=ner_use_noun_chunks,
            ner_batch_size=ner_batch_size,
            ner_n_process=ner_n_process,
            ner_allowed_labels=ner_allowed_labels,
            min_df=min_df,
            artifact_dir=entity_artifact_dir,
            force_rebuild=entity_force_rebuild,
        )
        self.doc_ids: List[str] = []
        self.doc_mat: np.ndarray = np.zeros((0, 1), dtype=np.float32)
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
        return ok

    def _save(self):
        if not (self.use_faiss and self.index):
            return
        self.faiss_helper.index = self.index
        self.faiss_helper.doc_ids = self.doc_ids
        self.faiss_helper.save()

    def build_index(self, docs: List[Document]) -> None:
        _log.info(f"🚀 Building Graph Index ({len(docs)} documentos)")
        self.doc_ids = [d.doc_id for d in docs]
        texts = [(d.title or "") + " " + (d.text or "") for d in docs]

        with log_time(_log, "Fit GraphVectorizer no corpus"):
            self.vec.fit_corpus(texts)

        with log_time(_log, "Encoding documents (Graph)"):
            vecs = [self.vec.encode_text(t) for t in texts]
            dim = self.vec.dim if vecs else 1
            self.doc_mat = np.stack(vecs, axis=0).astype(np.float32) if vecs else np.zeros((0, dim), dtype=np.float32)

        if self._try_load():
            _log.info(f"  ✓ Índice carregado do cache: {self._index_path}")
            return

        if self.use_faiss:
            with log_time(_log, "Construindo FAISS IndexFlatIP"):
                self.faiss_helper.build_from_matrix(self.doc_ids, self.doc_mat)
                self.index = self.faiss_helper.index
            _log.info(f"  ✓ FAISS IndexFlatIP: {self.index.ntotal} vetores, dim={self.vec.dim}")
            self._save()
        else:
            self.index = None
            _log.warning("  ⚠️  FAISS indisponível, usando NumPy fallback")

    def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        results: Dict[str, List[Tuple[str, float]]] = {}
        if len(self.doc_mat) == 0:
            return {q.query_id: [] for q in queries}

        for q in queries:
            qv = self.vec.encode_text(q.text).reshape(1, -1).astype(np.float32)
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
