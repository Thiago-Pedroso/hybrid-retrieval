"""
Factory for creating vectorizers from configuration.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from ..core.interfaces import AbstractVectorizer
from .dense_vectorizer import DenseVectorizer
from .tfidf_vectorizer import TFIDFVectorizer
from .bi_modal_vectorizer import BiModalVectorizer
from .tri_modal_vectorizer import TriModalVectorizer
from .graph_vectorizer import GraphVectorizer


def create_vectorizer(config: Dict[str, Any]) -> AbstractVectorizer:
    """Create a vectorizer from configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'type' key and type-specific config
        
    Returns:
        Vectorizer instance implementing AbstractVectorizer
        
    Examples:
        >>> config = {"type": "dense", "semantic": {"model": "sentence-transformers/all-MiniLM-L6-v2"}}
        >>> vec = create_vectorizer(config)
    """
    vec_type = config.get("type", "tri_modal").lower()
    
    if vec_type == "dense":
        semantic_cfg = config.get("semantic", {})
        return DenseVectorizer(
            model_name=semantic_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            device=semantic_cfg.get("device"),
            query_prefix=semantic_cfg.get("query_prefix", ""),
            doc_prefix=semantic_cfg.get("doc_prefix", ""),
        )
    
    elif vec_type == "tfidf":
        tfidf_cfg = config.get("tfidf", {})
        return TFIDFVectorizer(
            dim=tfidf_cfg.get("dim"),
            min_df=tfidf_cfg.get("min_df", 1),
            backend=tfidf_cfg.get("backend", "sklearn"),
        )
    
    elif vec_type == "bi_modal":
        semantic_cfg = config.get("semantic", {})
        tfidf_cfg = config.get("tfidf", {})
        return BiModalVectorizer(
            semantic_model_name=semantic_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            query_prefix=semantic_cfg.get("query_prefix", ""),
            doc_prefix=semantic_cfg.get("doc_prefix", ""),
            tfidf_dim=tfidf_cfg.get("dim", 1000),
            min_df=tfidf_cfg.get("min_df", 2),
            tfidf_backend=tfidf_cfg.get("backend", "sklearn"),
            tfidf_scale_multiplier=tfidf_cfg.get("scale_multiplier", 1.25),
            device=semantic_cfg.get("device"),
        )
    
    elif vec_type == "tri_modal":
        semantic_cfg = config.get("semantic", {})
        tfidf_cfg = config.get("tfidf", {})
        graph_cfg = config.get("graph", {})
        
        return TriModalVectorizer(
            tfidf_dim=tfidf_cfg.get("dim", 1000),
            min_df=tfidf_cfg.get("min_df", 2),
            semantic_model_name=semantic_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            tfidf_backend=tfidf_cfg.get("backend", "sklearn"),
            query_prefix=semantic_cfg.get("query_prefix", ""),
            doc_prefix=semantic_cfg.get("doc_prefix", ""),
            tfidf_scale_multiplier=tfidf_cfg.get("scale_multiplier", 1.25),
            graph_model_name=graph_cfg.get("model", "BAAI/bge-large-en-v1.5"),
            ner_backend=graph_cfg.get("ner_backend", "scispacy"),
            ner_model=graph_cfg.get("ner_model"),
            ner_use_noun_chunks=graph_cfg.get("ner_use_noun_chunks", True),
            ner_batch_size=graph_cfg.get("ner_batch_size", 128),
            ner_n_process=graph_cfg.get("ner_n_process", 4),
            ner_allowed_labels=graph_cfg.get("ner_allowed_labels"),
            entity_min_df=graph_cfg.get("entity_min_df"),
            entity_max_entities_per_text=graph_cfg.get("entity_max_entities_per_text", 128),
            entity_artifact_dir=graph_cfg.get("entity_artifact_dir"),
            entity_force_rebuild=graph_cfg.get("entity_force_rebuild", False),
            device=semantic_cfg.get("device"),
        )
    
    elif vec_type == "graph":
        graph_cfg = config.get("graph", {})
        return GraphVectorizer(
            graph_model_name=graph_cfg.get("model", "BAAI/bge-large-en-v1.5"),
            device=graph_cfg.get("device"),
            ner_backend=graph_cfg.get("ner_backend", "scispacy"),
            ner_model=graph_cfg.get("ner_model"),
            ner_use_noun_chunks=graph_cfg.get("ner_use_noun_chunks", True),
            ner_batch_size=graph_cfg.get("ner_batch_size", 128),
            ner_n_process=graph_cfg.get("ner_n_process", 4),
            ner_allowed_labels=graph_cfg.get("ner_allowed_labels"),
            min_df=graph_cfg.get("min_df", 2),
            artifact_dir=graph_cfg.get("artifact_dir"),
            force_rebuild=graph_cfg.get("force_rebuild", False),
        )
    
    else:
        raise ValueError(
            f"Unknown vectorizer type: {vec_type}. "
            f"Available: dense, tfidf, bi_modal, tri_modal, graph"
        )