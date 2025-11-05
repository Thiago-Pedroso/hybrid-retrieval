"""
Factory for creating indexes from configuration.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from ..core.interfaces import AbstractIndex, AbstractVectorizer
from .hybrid_index import HybridIndex


def create_index(
    config: Dict[str, Any],
    vectorizer: AbstractVectorizer,
) -> AbstractIndex:
    """Create an index from configuration and vectorizer.
    
    Args:
        config: Configuration dictionary with index settings
        vectorizer: Vectorizer instance to use for encoding
        
    Returns:
        Index instance implementing AbstractIndex
        
    Examples:
        >>> config = {"type": "faiss", "factory": None, "metric": "ip"}
        >>> index = create_index(config, vectorizer)
    """
    index_type = config.get("type", "faiss").lower()
    
    if index_type == "faiss":
        return HybridIndex(
            vectorizer=vectorizer,
            faiss_factory=config.get("factory"),
            faiss_metric=config.get("metric", "ip"),
            faiss_nprobe=config.get("nprobe"),
            faiss_train_size=config.get("train_size", 0),
            artifact_dir=config.get("artifact_dir"),
            index_name=config.get("index_name", "index"),
        )
    
    elif index_type == "numpy":
        # NumPy-based index (fallback when FAISS not available)
        # For now, we'll use HybridIndex with FAISS disabled
        # In the future, could create a dedicated NumPyIndex class
        return HybridIndex(
            vectorizer=vectorizer,
            faiss_factory=None,  # Forces NumPy fallback
            faiss_metric=config.get("metric", "ip"),
            faiss_nprobe=None,
            faiss_train_size=0,
            artifact_dir=config.get("artifact_dir"),
            index_name=config.get("index_name", "index"),
        )
    
    else:
        raise ValueError(
            f"Unknown index type: {index_type}. Available: faiss, numpy"
        )

