from __future__ import annotations
import numpy as np
from typing import Dict, Iterable, Optional
from ..encoders.encoders import HFSemanticEncoder, TfidfEncoder, l2norm
from ..utils.logging import get_logger, log_time
from ..core.interfaces import AbstractVectorizer

_log = get_logger("bi_modal.vectorizer")

class BiModalVectorizer(AbstractVectorizer):
    """
    Vectorizer híbrido bi-modal: semântico (s) + lexical (t).
    Concatena [s, t'] onde t' = t̂ × √(D_s / D_t) (escalonamento proporcional à raiz).
    """
    def __init__(self,
                 # Semântico
                 semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 query_prefix: str = "",
                 doc_prefix: str = "",
                 # TF-IDF
                 tfidf_dim: Optional[int] = 1000,
                 min_df: int = 2,
                 tfidf_backend: str = "sklearn",
                 tfidf_scale_multiplier: float = 1.25,
                 # Device
                 device: Optional[str] = None):
        
        # Encoder semântico
        self.semantic = HFSemanticEncoder(
            model_name=semantic_model_name,
            device=device,
            query_prefix=query_prefix,
            doc_prefix=doc_prefix,
        )
        
        # Encoder TF-IDF
        self.tfidf = TfidfEncoder(
            dim=tfidf_dim,
            min_df=min_df,
            backend=tfidf_backend
        )
        
        self.tfidf_scale_multiplier = tfidf_scale_multiplier
        self.slice_dims = {}
        self.fitted = False

    def fit_corpus(self, docs_texts: Iterable[str]):
        docs_texts = list(docs_texts)
        _log.info(f"🔧 Fitting BiModal vectorizer com {len(docs_texts)} documentos")
        
        # TF-IDF
        with log_time(_log, "Fit TF-IDF"):
            self.tfidf.fit(docs_texts)
            tdim = self.tfidf.vocab_size
        _log.info(f"  ✓ TF-IDF fitted: vocab_size={tdim}")
        
        # Semântico (não precisa fit, apenas detecta dim)
        sdim = int(self.semantic.dim or 384)
        
        self.slice_dims = {"s": sdim, "t": tdim}
        total_dim = sdim + tdim
        _log.info(f"  ✓ Dimensões: semantic={sdim}, tfidf={tdim}, total={total_dim}")
        self.fitted = True

    def encode_text(self, text: str, is_query: bool = False) -> Dict[str, np.ndarray]:
        assert self.fitted, "Chame fit_corpus() antes de encode_text()"
        _log.debug(f"Encoding {'query' if is_query else 'document'}: {text[:100]}...")
        
        # Encode s e t
        s = self.semantic.encode_text(text, is_query=is_query)
        t = self.tfidf.encode_text(text) if self.slice_dims["t"] > 0 else np.zeros(0, dtype=np.float32)
        
        # Escalonamento: t' = t̂ × √(D_s / D_t) × multiplier
        # Multiplier padrão 1.25× foi otimizado empiricamente para melhor balanceamento
        D_s = self.slice_dims["s"]
        D_t = self.slice_dims["t"]
        scale_factor = 1.0
        if D_t > 0 and t.size > 0:
            base_scale = np.sqrt(float(D_s) / float(D_t))
            scale_factor = base_scale * self.tfidf_scale_multiplier
            t = t * scale_factor
        
        _log.debug(f"Encoded vectors: s_norm={np.linalg.norm(s):.3f}, t_norm={np.linalg.norm(t):.3f}, tfidf_scale={scale_factor:.4f}")
        return {"s": s, "t": t}

    def concat(self, parts: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatena [s, t] e normaliza L2."""
        v = np.concatenate([parts["s"], parts["t"]]).astype(np.float32)
        return l2norm(v)

    @property
    def total_dim(self) -> int:
        """Total dimension of concatenated vector."""
        return int(self.slice_dims.get("s", 0) + self.slice_dims.get("t", 0))

