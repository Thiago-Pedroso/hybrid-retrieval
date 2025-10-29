from __future__ import annotations
import numpy as np
from typing import Optional
from ..encoders.encoders import HFSemanticEncoder, l2norm

class DenseVectorizer:
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 query_prefix: str = "",
                 doc_prefix: str = ""):
        self.encoder = HFSemanticEncoder(
            model_name=model_name,
            device=device,
            query_prefix=query_prefix,
            doc_prefix=doc_prefix,
        )
        self.dim = int(self.encoder.dim or 384)

    def encode_query(self, text: str) -> np.ndarray:
        return l2norm(self.encoder.encode_text(text, is_query=True))

    def encode_doc(self, text: str) -> np.ndarray:
        return l2norm(self.encoder.encode_text(text, is_query=False))