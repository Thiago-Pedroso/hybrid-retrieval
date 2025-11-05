from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Iterable
from ..encoders.encoders import HFSemanticEncoder, OpenAISemanticEncoder, l2norm
from ..core.interfaces import AbstractVectorizer


class DenseVectorizer(AbstractVectorizer):
    """Dense vectorizer using semantic embeddings only.
    Supports both HuggingFace models and OpenAI API.
    """

    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 query_prefix: str = "",
                 doc_prefix: str = "",
                 provider: Optional[str] = None,
                 api_key: Optional[str] = None):
        """
        Args:
            model_name: Model name (HuggingFace path or OpenAI model name)
            device: Device for HuggingFace models (ignored for OpenAI)
            query_prefix: Prefix for queries
            doc_prefix: Prefix for documents
            provider: "openai" or "huggingface" (auto-detected if None)
            api_key: OpenAI API key (optional, can use .env)
        """
        # Auto-detect provider based on model name
        if provider is None:
            if model_name.startswith("text-embedding") or "openai" in model_name.lower():
                provider = "openai"
            else:
                provider = "huggingface"
        
        if provider == "openai":
            self.encoder = OpenAISemanticEncoder(
                model_name=model_name,
                query_prefix=query_prefix,
                doc_prefix=doc_prefix,
                api_key=api_key,
            )
            self._dim = int(self.encoder.dim)
        else:
            self.encoder = HFSemanticEncoder(
                model_name=model_name,
                device=device,
                query_prefix=query_prefix,
                doc_prefix=doc_prefix,
            )
            self._dim = int(self.encoder.dim or 384)
        
        self._provider = provider
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
    