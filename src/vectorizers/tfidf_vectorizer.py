from __future__ import annotations
import numpy as np
from typing import Iterable, Dict
from ..encoders.encoders import TfidfEncoder, l2norm
from ..utils.logging import get_logger, log_time
from ..core.interfaces import AbstractVectorizer

_log = get_logger("tfidf.vectorizer")

class TFIDFVectorizer(AbstractVectorizer):
    """TF-IDF only vectorizer."""
    
    def __init__(self, dim: int = None, min_df: int = 1, backend: str = "sklearn"):
        self.encoder = TfidfEncoder(dim=dim, min_df=min_df, backend=backend)
        self._fitted = False

    def fit_corpus(self, docs_texts: Iterable[str]) -> None:
        """Fit TF-IDF on corpus."""
        docs = list(docs_texts)
        with log_time(_log, "Fit TF-IDF"):
            self.encoder.fit(docs)
        self._fitted = True
        _log.info(f"✓ TF-IDF fitted: vocab_size={self.encoder.vocab_size}")

    def encode_text(self, text: str, is_query: bool = False) -> Dict[str, np.ndarray]:
        """Encode text as TF-IDF vector."""
        assert self._fitted, "Chame fit_corpus() antes de encode_text()"
        v = l2norm(self.encoder.encode_text(text))
        return {"t": v}

    def concat(self, parts: Dict[str, np.ndarray]) -> np.ndarray:
        """Return TF-IDF vector directly (already single vector)."""
        return parts["t"]

    @property
    def total_dim(self) -> int:
        """Total dimension (TF-IDF only)."""
        return int(self.encoder.dim)
    