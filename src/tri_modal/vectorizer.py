# src/tri_modal/vectorizer.py
from __future__ import annotations
import numpy as np
from typing import Dict, Iterable, Optional
from .encoders import HFSemanticEncoder, _StubSemanticEncoder, TfidfEncoder, l2norm
from .entity_encoder import EntityEncoderReal, NERConfig

class TriModalVectorizer:
    """
    Monta [ s ; t ; g ] com L2 por fatia e L2 global.

    Parâmetros principais:
      - semantic_backend: "hf" | "stub"
      - semantic_model_name: ex. "sentence-transformers/all-MiniLM-L6-v2" ou "BAAI/bge-large-en-v1.5"
      - tfidf_backend: "sklearn" | "pyserini"
      - graph_model_name: modelo HF para embutir entidades (default: BGE-Large)
      - ner_backend/model: qual pipeline spaCy/scispaCy usar para NER
      - query_prefix/doc_prefix: úteis para BGE (ex.: "query: ", "passage: ")
    """
    def __init__(self,
                 sem_dim: Optional[int] = None,                 # usado apenas no stub
                 tfidf_dim: int = 1000,
                 seed: int = 42,
                 min_df: int = 2,
                 semantic_backend: str = "hf",
                 semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 tfidf_backend: str = "sklearn",
                 query_prefix: str = "",
                 doc_prefix: str = "",
                 # ==== novos (entidades):
                 graph_model_name: str = "BAAI/bge-large-en-v1.5",
                 ner_backend: str = "scispacy",
                 ner_model: Optional[str] = None,
                 ner_use_noun_chunks: bool = True,
                 ner_batch_size: int = 64,
                 ner_n_process: int = 1,
                 device: Optional[str] = None):
        # SEMÂNTICO (s)
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

        # LEXICAL (t)
        self.tfidf = TfidfEncoder(dim=tfidf_dim, min_df=min_df, backend=tfidf_backend)

        # ENTIDADES (g)
        ner_cfg = NERConfig(
            backend=ner_backend,
            model=ner_model,
            use_noun_chunks=ner_use_noun_chunks,
            batch_size=ner_batch_size,
            n_process=ner_n_process,
        )
        self.entities = EntityEncoderReal(
            graph_model_name=graph_model_name,
            device=device,
            ner=ner_cfg,
            min_df=min_df,
        )

        self.slice_dims = {}  # {"s": sem_out_dim, "t": ?, "g": ent_dim}
        self.fitted = False

    def fit_corpus(self, docs_texts: Iterable[str]):
        docs_texts = list(docs_texts)

        # TF-IDF
        self.tfidf.fit(docs_texts)
        tdim = self.tfidf.vocab_size

        # ENTIDADES (DF/IDF no corpus)
        self.entities.fit(docs_texts)
        gdim = self.entities.dim

        sdim = int(self.semantic.dim or 384)
        self.slice_dims = {"s": sdim, "t": tdim, "g": gdim}
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
        s = l2norm(self._encode_semantic(text, is_query=is_query))
        t = l2norm(self._encode_tfidf(text))
        g = l2norm(self._encode_entities(text))
        return {"s": s, "t": t, "g": g}

    def concat(self, parts: Dict[str, np.ndarray]) -> np.ndarray:
        v = np.concatenate([parts["s"], parts["t"], parts["g"]]).astype(np.float32)
        return l2norm(v)
