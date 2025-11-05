"""
Compatibility wrappers for retrievers.
Maintains old API for backward compatibility.
"""

from __future__ import annotations
from typing import Optional
from ..retrievers.dense_faiss import DenseFaiss


class DenseFaissStub(DenseFaiss):
    """
    Compatibility wrapper for DenseFaissStub.
    
    Legacy API: DenseFaissStub(dim=384)
    New API: DenseFaiss(model_name="...")
    
    This wrapper maintains the old API while using the new implementation.
    """
    
    def __init__(self, dim: int = 384, **kwargs):
        """
        Initialize with dimension (legacy API).
        
        Args:
            dim: Dimension (legacy parameter, maps to appropriate model)
            **kwargs: Additional arguments passed to DenseFaiss
        """
        # Map dimension to model (legacy behavior)
        # Common dimensions: 384 (MiniLM), 768 (BERT-base), 1024 (BGE-large)
        model_map = {
            384: "sentence-transformers/all-MiniLM-L6-v2",
            768: "sentence-transformers/all-mpnet-base-v2",
            1024: "BAAI/bge-large-en-v1.5",
        }
        
        model_name = model_map.get(dim, "sentence-transformers/all-MiniLM-L6-v2")
        
        # Remove 'dim' from kwargs if present (shouldn't be in new API)
        kwargs.pop("dim", None)
        
        # Initialize parent with model
        super().__init__(model_name=model_name, **kwargs)

