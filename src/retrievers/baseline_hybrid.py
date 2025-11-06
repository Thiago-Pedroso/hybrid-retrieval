"""
Baseline Hybrid Retriever with fixed alpha.
Same structure as DAT but uses fixed alpha instead of LLM judge.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from ..datasets.schema import Document, Query
from .base import AbstractRetriever
from .bm25_basic import BM25Basic
from .dense_faiss import DenseFaiss
from ..fusion.normalization import normalize_scores_minmax
from ..fusion.strategies import DATLinearFusion
from ..utils.logging import get_logger, log_time

_log = get_logger("retriever.baseline_hybrid")


class BaselineHybridRetriever(AbstractRetriever):
    """
    Baseline Hybrid Retriever with fixed alpha.
    Same structure as DAT but uses fixed alpha instead of LLM judge.
    
    Used for grid search: α ∈ {0.0, 0.1, 0.2, ..., 1.0}
    """
    
    def __init__(
        self,
        bm25_retriever: BM25Basic,
        dense_retriever: DenseFaiss,
        alpha: float,
        top_k: int = 20,
    ):
        """
        Args:
            bm25_retriever: BM25 retriever instance
            dense_retriever: Dense retriever instance
            alpha: Fixed alpha value for fusion (0.0 to 1.0)
            top_k: Number of top results to retrieve (default: 20)
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.alpha = alpha
        self.top_k = top_k
    
    def build_index(self, docs: List[Document]) -> None:
        """Build indices for both retrievers."""
        _log.info(f"🚀 Building Baseline Hybrid Index (α={self.alpha}, {len(docs)} documentos)")
        
        # Build BM25 index
        with log_time(_log, "Build BM25 index"):
            self.bm25_retriever.build_index(docs)
        
        # Build Dense index
        with log_time(_log, "Build Dense index"):
            self.dense_retriever.build_index(docs)
        
        _log.info("✅ Baseline Hybrid Index built successfully")
    
    def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieve documents using baseline hybrid approach with fixed alpha.
        
        Args:
            queries: List of queries
            k: Number of results to return (default: 10)
        
        Returns:
            Dictionary mapping query_id to list of (doc_id, score) tuples
        """
        _log.info(f"🔍 Baseline Hybrid Retrieval (α={self.alpha}): {len(queries)} queries, top-{k}")
        
        results: Dict[str, List[Tuple[str, float]]] = {}
        
        for q in queries:
            # Step 1: Retrieve top-K from both methods
            bm25_results = self.bm25_retriever.retrieve([q], k=self.top_k)
            dense_results = self.dense_retriever.retrieve([q], k=self.top_k)
            
            bm25_topk = bm25_results.get(q.query_id, [])
            dense_topk = dense_results.get(q.query_id, [])
            
            if not bm25_topk or not dense_topk:
                # Fallback: return empty or best available
                if bm25_topk:
                    results[q.query_id] = bm25_topk[:k]
                elif dense_topk:
                    results[q.query_id] = dense_topk[:k]
                else:
                    results[q.query_id] = []
                continue
            
            # Step 2: Normalize scores separately (min-max per method)
            bm25_normalized = normalize_scores_minmax(bm25_topk)
            dense_normalized = normalize_scores_minmax(dense_topk)
            
            # Step 3: Fuse using DATLinearFusion with fixed alpha
            fusion_strategy = DATLinearFusion(alpha=self.alpha)
            fused_results = fusion_strategy.fuse(
                query=q.text,
                results_list=[
                    {q.query_id: bm25_normalized},
                    {q.query_id: dense_normalized},
                ],
            )
            
            # Step 4: Return top-k
            results[q.query_id] = fused_results[:k]
        
        _log.info(f"✅ Baseline Hybrid Retrieval completed: {len(results)} queries")
        return results

