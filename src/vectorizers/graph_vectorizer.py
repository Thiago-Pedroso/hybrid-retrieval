from __future__ import annotations
import numpy as np
from typing import Iterable, Optional, List
from pathlib import Path
from ..encoders.entity_encoder import EntityEncoderReal, NERConfig, CacheConfig
from ..utils.logging import get_logger, log_time

_log = get_logger("graph.vectorizer")

class GraphVectorizer:
    def __init__(self,
                 graph_model_name: str = "BAAI/bge-large-en-v1.5",
                 device: Optional[str] = None,
                 ner_backend: str = "scispacy",
                 ner_model: Optional[str] = None,
                 ner_use_noun_chunks: bool = True,
                 ner_batch_size: int = 64,
                 ner_n_process: int = 1,
                 ner_allowed_labels: Optional[List[str]] = None,
                 min_df: int = 2,
                 artifact_dir: Optional[str] = None,
                 force_rebuild: bool = False):
        ner_cfg = NERConfig(
            backend=ner_backend,
            model=ner_model,
            use_noun_chunks=ner_use_noun_chunks,
            batch_size=ner_batch_size,
            n_process=ner_n_process,
            allowed_labels=ner_allowed_labels,
        )
        cache_cfg = CacheConfig(
            artifact_dir=Path(artifact_dir) if artifact_dir else None,
            force_rebuild=force_rebuild,
        )
        self.encoder = EntityEncoderReal(
            graph_model_name=graph_model_name,
            device=device,
            ner=ner_cfg,
            min_df=min_df,
            cache=cache_cfg,
        )
        self._fitted = False

    def fit_corpus(self, docs_texts: Iterable[str]):
        docs = list(docs_texts)
        with log_time(_log, "Fit Graph (NER + IDF)"):
            self.encoder.fit(docs)
        self._fitted = True
        _log.info(f"✓ Graph fitted: dim={self.encoder.dim}, ents={len(self.encoder.ent2idf)}")

    def encode_text(self, text: str) -> np.ndarray:
        assert self._fitted, "Chame fit_corpus() antes de encode_text()"
        return self.encoder.encode_text(text)

    @property
    def dim(self) -> int:
        return int(self.encoder.dim)