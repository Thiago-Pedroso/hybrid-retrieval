"""
Factory for creating retrievers from configuration.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from ..core.interfaces import AbstractRetriever
from ..vectorizers.factory import create_vectorizer
from ..indexes.factory import create_index
from ..fusion.factory import create_weight_policy, create_reranker, create_fusion_strategy_from_config, create_llm_judge, create_dat_weight_policy
from .hybrid_faiss import HybridRetriever
from .dense_faiss import DenseFaiss
from .tfidf_faiss import TFIDFRetriever
from .graph_faiss import GraphRetriever
from .bm25_basic import BM25Basic
from .dat_hybrid import DATHybridRetriever
from .baseline_hybrid import BaselineHybridRetriever


def create_retriever(config: Dict[str, Any]) -> AbstractRetriever:
    """Create a retriever from configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'type' and component configs
        
    Returns:
        Retriever instance implementing AbstractRetriever
        
    Examples:
        >>> config = {
        ...     "type": "hybrid",
        ...     "vectorizer": {"type": "tri_modal", ...},
        ...     "fusion": {"strategy": "weighted_cosine", "policy": "heuristic"},
        ...     "reranker": {"type": "tri_modal", "topk_first": 150},
        ...     "index": {"type": "faiss", "metric": "ip"}
        ... }
        >>> retriever = create_retriever(config)
    """
    retriever_type = config.get("type", "hybrid").lower()
    
    if retriever_type == "hybrid":
        # Create components
        vectorizer_config = config.get("vectorizer", {})
        vectorizer = create_vectorizer(vectorizer_config)
        
        index_config = config.get("index", {})
        index = create_index(index_config, vectorizer)
        
        fusion_config = config.get("fusion", {})
        weight_policy = create_weight_policy(fusion_config)
        
        reranker_config = config.get("reranker", {})
        reranker = create_reranker(
            reranker_config.get("type", "tri_modal"),
            vectorizer
        )
        
        topk_first = reranker_config.get("topk_first", 150)
        
        # Create HybridRetriever with components
        return HybridRetriever(
            vectorizer=vectorizer,
            index=index,
            reranker=reranker,
            weight_policy=weight_policy,
            topk_first=topk_first,
        )
    
    elif retriever_type == "dense":
        semantic_cfg = config.get("vectorizer", {}).get("semantic", {}) or config.get("semantic", {})
        return DenseFaiss(
            model_name=semantic_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            device=semantic_cfg.get("device"),
            query_prefix=semantic_cfg.get("query_prefix", ""),
            doc_prefix=semantic_cfg.get("doc_prefix", ""),
            provider=semantic_cfg.get("provider"),  # "openai" or "huggingface"
            api_key=semantic_cfg.get("api_key"),  # Optional, can use .env
            artifact_dir=config.get("index", {}).get("artifact_dir"),
            index_name=config.get("index", {}).get("index_name", "dense.index"),
        )
    
    elif retriever_type == "tfidf":
        tfidf_cfg = config.get("vectorizer", {}).get("tfidf", {}) or config.get("tfidf", {})
        return TFIDFRetriever(
            dim=tfidf_cfg.get("dim"),
            min_df=tfidf_cfg.get("min_df", 1),
            backend=tfidf_cfg.get("backend", "sklearn"),
            artifact_dir=config.get("index", {}).get("artifact_dir"),
            index_name=config.get("index", {}).get("index_name", "tfidf.index"),
        )
    
    elif retriever_type == "graph":
        graph_cfg = config.get("vectorizer", {}).get("graph", {}) or config.get("graph", {})
        return GraphRetriever(
            graph_model_name=graph_cfg.get("model", "BAAI/bge-large-en-v1.5"),
            device=graph_cfg.get("device"),
            ner_backend=graph_cfg.get("ner_backend", "scispacy"),
            ner_model=graph_cfg.get("ner_model"),
            ner_use_noun_chunks=graph_cfg.get("ner_use_noun_chunks", True),
            ner_batch_size=graph_cfg.get("ner_batch_size", 128),
            ner_n_process=graph_cfg.get("ner_n_process", 4),
            ner_allowed_labels=graph_cfg.get("ner_allowed_labels"),
            min_df=graph_cfg.get("min_df", 2),
            artifact_dir=config.get("index", {}).get("artifact_dir"),
            index_name=config.get("index", {}).get("index_name", "graph.index"),
            entity_artifact_dir=graph_cfg.get("entity_artifact_dir"),
            entity_force_rebuild=graph_cfg.get("entity_force_rebuild", False),
        )
    
    elif retriever_type == "bm25":
        k1 = config.get("k1", 0.9)
        b = config.get("b", 0.4)
        return BM25Basic(k1=k1, b=b)
    
    elif retriever_type == "dat_hybrid":
        # Create BM25 retriever
        bm25_config = config.get("bm25", {})
        bm25_retriever = BM25Basic(
            k1=bm25_config.get("k1", 0.9),
            b=bm25_config.get("b", 0.4),
        )
        
        # Create Dense retriever
        dense_config = config.get("dense", {})
        semantic_cfg = dense_config.get("semantic", {}) or dense_config
        dense_retriever = DenseFaiss(
            model_name=semantic_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            device=semantic_cfg.get("device"),
            query_prefix=semantic_cfg.get("query_prefix", ""),
            doc_prefix=semantic_cfg.get("doc_prefix", ""),
            provider=semantic_cfg.get("provider"),
            api_key=semantic_cfg.get("api_key"),
            artifact_dir=dense_config.get("index", {}).get("artifact_dir"),
            index_name=dense_config.get("index", {}).get("index_name", "dense.index"),
        )
        
        # Create LLM judge
        fusion_config = config.get("fusion", {})
        llm_judge_config = fusion_config.get("llm_judge", {})
        llm_judge = create_llm_judge(llm_judge_config)
        
        # Create DAT weight policy
        weight_policy = create_dat_weight_policy()
        
        # Top-K for retrieval
        top_k = fusion_config.get("top_k", 20)
        
        return DATHybridRetriever(
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
            llm_judge=llm_judge,
            weight_policy=weight_policy,
            top_k=top_k,
        )
    
    elif retriever_type == "baseline_hybrid":
        # Create BM25 retriever
        bm25_config = config.get("bm25", {})
        bm25_retriever = BM25Basic(
            k1=bm25_config.get("k1", 0.9),
            b=bm25_config.get("b", 0.4),
        )
        
        # Create Dense retriever
        dense_config = config.get("dense", {})
        semantic_cfg = dense_config.get("semantic", {}) or dense_config
        dense_retriever = DenseFaiss(
            model_name=semantic_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            device=semantic_cfg.get("device"),
            query_prefix=semantic_cfg.get("query_prefix", ""),
            doc_prefix=semantic_cfg.get("doc_prefix", ""),
            provider=semantic_cfg.get("provider"),
            api_key=semantic_cfg.get("api_key"),
            artifact_dir=dense_config.get("index", {}).get("artifact_dir"),
            index_name=dense_config.get("index", {}).get("index_name", "dense.index"),
        )
        
        # Fixed alpha
        fusion_config = config.get("fusion", {})
        alpha = fusion_config.get("alpha", 0.6)
        if alpha is None:
            raise ValueError("baseline_hybrid requires 'alpha' in fusion config")
        
        # Top-K for retrieval
        top_k = fusion_config.get("top_k", 20)
        
        return BaselineHybridRetriever(
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
            alpha=alpha,
            top_k=top_k,
        )
    
    else:
        raise ValueError(
            f"Unknown retriever type: {retriever_type}. "
            f"Available: hybrid, dense, tfidf, graph, bm25, dat_hybrid, baseline_hybrid"
        )

