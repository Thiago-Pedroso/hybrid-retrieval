from __future__ import annotations
import numpy as np
from typing import Dict, Iterable, Optional, List
from pathlib import Path
from .encoders import HFSemanticEncoder, _StubSemanticEncoder, TfidfEncoder, l2norm
from .entity_encoder import EntityEncoderReal, NERConfig, CacheConfig
from ..utils.logging import get_logger, log_time

_log = get_logger("tri_modal.vectorizer")

class TriModalVectorizer:
    def __init__(self,
                 sem_dim: Optional[int] = None,
                 tfidf_dim: int = 1000,
                 seed: int = 42,
                 min_df: int = 2,
                 semantic_backend: str = "hf",
                 semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 tfidf_backend: str = "sklearn",
                 query_prefix: str = "",
                 doc_prefix: str = "",
                 # ENTIDADES
                 graph_model_name: str = "BAAI/bge-large-en-v1.5",
                 ner_backend: str = "scispacy",
                 ner_model: Optional[str] = None,
                 ner_use_noun_chunks: bool = True,
                 ner_batch_size: int = 64,
                 ner_n_process: int = 1,
                 ner_allowed_labels: Optional[List[str]] = None,
                 entity_artifact_dir: Optional[str] = None,
                 entity_force_rebuild: bool = False,
                 device: Optional[str] = None):
        # semântico (s)
        if semantic_backend == "hf":
            self.semantic = HFSemanticEncoder(
                model_name=semantic_model_name,
                device=device,
                query_prefix=query_prefix,
                doc_prefix=doc_prefix,
            )
            sem_out_dim = int(self.semantic.dim or 384)
        else:
            self.semantic = _StubSemanticEncoder(dim=sem_dim or 384, seed=seed)
            sem_out_dim = self.semantic.dim

        # lexical (t)
        self.tfidf = TfidfEncoder(dim=tfidf_dim, min_df=min_df, backend=tfidf_backend)

        # entidades (g)
        ner_cfg = NERConfig(
            backend=ner_backend,
            model=ner_model,
            use_noun_chunks=ner_use_noun_chunks,
            batch_size=ner_batch_size,
            n_process=ner_n_process,
            allowed_labels=ner_allowed_labels,
        )
        cache_cfg = CacheConfig(
            artifact_dir=Path(entity_artifact_dir) if entity_artifact_dir else None,
            force_rebuild=entity_force_rebuild,
        )
        self.entities = EntityEncoderReal(
            graph_model_name=graph_model_name,
            device=device,
            ner=ner_cfg,
            min_df=min_df,
            cache=cache_cfg,
        )

        self.slice_dims = {}
        self.fitted = False

    def fit_corpus(self, docs_texts: Iterable[str]):
        docs_texts = list(docs_texts)
        _log.info(f"🔧 Fitting TriModal vectorizer com {len(docs_texts)} documentos")
        
        # TF-IDF
        with log_time(_log, "Fit TF-IDF"):
            self.tfidf.fit(docs_texts)
            tdim = self.tfidf.vocab_size
        _log.info(f"  ✓ TF-IDF fitted: vocab_size={tdim}")
        
        # ENTIDADES
        with log_time(_log, "Fit Entity Encoder (NER + IDF)"):
            self.entities.fit(docs_texts)
            gdim = self.entities.dim
        _log.info(f"  ✓ Entity Encoder fitted: dim={gdim}, vocab={len(self.entities.ent2idf)} entidades")
        
        # s
        sdim = int(self.semantic.dim or 384)
        self.slice_dims = {"s": sdim, "t": tdim, "g": gdim}
        total_dim = sdim + tdim + gdim
        _log.info(f"  ✓ Dimensões: semantic={sdim}, tfidf={tdim}, entities={gdim}, total={total_dim}")
        self.fitted = True

    def _encode_semantic(self, text: str, is_query: bool) -> np.ndarray:
        return self.semantic.encode_text(text, is_query=is_query)

    def _encode_tfidf(self, text: str) -> np.ndarray:
        if self.slice_dims["t"] == 0:
            return np.zeros(0, dtype=np.float32)
        return self.tfidf.encode_text(text)

    def _encode_entities(self, text: str) -> np.ndarray:
        return self.entities.encode_text(text)

    def encode_text(self, text: str, is_query: bool = False) -> Dict[str, np.ndarray]:
        assert self.fitted, "Chame fit_corpus() antes de encode_text()"
        _log.debug(f"Encoding {'query' if is_query else 'document'}: {text[:100]}...")
        s = l2norm(self._encode_semantic(text, is_query=is_query))
        t = l2norm(self._encode_tfidf(text))
        g = l2norm(self._encode_entities(text))
        _log.debug(f"Encoded vectors: s_norm={np.linalg.norm(s):.3f}, t_norm={np.linalg.norm(t):.3f}, g_norm={np.linalg.norm(g):.3f}")
        return {"s": s, "t": t, "g": g}

    def concat(self, parts: Dict[str, np.ndarray]) -> np.ndarray:
        v = np.concatenate([parts["s"], parts["t"], parts["g"]]).astype(np.float32)
        return l2norm(v)