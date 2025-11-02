from __future__ import annotations
import numpy as np
from typing import Iterable
from ..encoders.encoders import TfidfEncoder, l2norm
from ..utils.logging import get_logger, log_time

_log = get_logger("tfidf.vectorizer")

class TFIDFVectorizer:
    def __init__(self, dim: int = None, min_df: int = 1, backend: str = "sklearn"):
        self.encoder = TfidfEncoder(dim=dim, min_df=min_df, backend=backend)
        self._fitted = False

    def fit_corpus(self, docs_texts: Iterable[str]):
        docs = list(docs_texts)
        with log_time(_log, "Fit TF-IDF"):
            self.encoder.fit(docs)
        self._fitted = True
        _log.info(f"✓ TF-IDF fitted: vocab_size={self.encoder.vocab_size}")

    def encode_text(self, text: str) -> np.ndarray:
        assert self._fitted, "Chame fit_corpus() antes de encode_text()"
        v = self.encoder.encode_text(text)
        return l2norm(v)

    @property
    def dim(self) -> int:
        return int(self.encoder.dim)