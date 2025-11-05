from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Iterable
from ..encoders.encoders import HFSemanticEncoder, l2norm
from ..core.interfaces import AbstractVectorizer


class DenseVectorizer(AbstractVectorizer):
    """Dense vectorizer using semantic embeddings only."""
    
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
        self._dim = int(self.encoder.dim or 384)
        self._fitted = True  # Semantic encoder doesn't need fitting
    
    def fit_corpus(self, docs_texts: Iterable[str]) -> None:
        """No-op for dense vectorizer (semantic encoder doesn't need fitting)."""
        self._fitted = True
    
    def encode_text(self, text: str, is_query: bool = False) -> Dict[str, np.ndarray]:
        """Encode text as semantic vector."""
        vec = l2norm(self.encoder.encode_text(text, is_query=is_query))
        return {"s": vec}
    
    def concat(self, parts: Dict[str, np.ndarray]) -> np.ndarray:
        """Return semantic vector directly (already single vector)."""
        return parts["s"]
    
    @property
    def total_dim(self) -> int:
        """Total dimension (semantic only)."""
        return self._dim
    
    # Backward compatibility methods
    def encode_query(self, text: str) -> np.ndarray:
        """Backward compatibility: encode query."""
        return self.encode_text(text, is_query=True)["s"]
    
    def encode_doc(self, text: str) -> np.ndarray:
        """Backward compatibility: encode document."""
        return self.encode_text(text, is_query=False)["s"]