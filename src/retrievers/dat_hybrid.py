"""
DAT (Dynamic Alpha Tuning) Hybrid Retriever.
Combines BM25 and Dense retrieval with adaptive weights computed via LLM judge.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from ..datasets.schema import Document, Query
from .base import AbstractRetriever
from .bm25_basic import BM25Basic
from .dense_faiss import DenseFaiss
from ..fusion.llm_judge import LLMJudge
from ..fusion.weighting import DATWeightPolicy
from ..fusion.normalization import normalize_scores_minmax
from ..fusion.strategies import DATLinearFusion
from ..utils.logging import get_logger, log_time

_log = get_logger("retriever.dat_hybrid")


class DATHybridRetriever(AbstractRetriever):
    """
    DAT Hybrid Retriever that combines BM25 and Dense retrieval with adaptive alpha.
    
    Pipeline:
    1. Retrieve top-K from BM25 and Dense
    2. Extract top-1 from each method
    3. Evaluate effectiveness via LLM judge (both together)
    4. Normalize judge scores: e^d = e_d/5, e^b = e_b/5
    5. Compute α(q) with edge cases
    6. Normalize BM25@K and Dense@K scores separately (min-max)
    7. Union candidates (BM25@K ∪ Dense@K)
    8. Assign score=0 for missing documents
    9. Fuse with linear combination
    10. Sort by R(q,d) with stable tie-breaking
    """
    
    def __init__(
        self,
        bm25_retriever: BM25Basic,
        dense_retriever: DenseFaiss,
        llm_judge: LLMJudge,
        weight_policy: DATWeightPolicy,
        top_k: int = 20,
        doc_map: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            bm25_retriever: BM25 retriever instance
            dense_retriever: Dense retriever instance
            llm_judge: LLM judge instance
            weight_policy: DAT weight policy instance
            top_k: Number of top results to retrieve (default: 20)
            doc_map: Optional mapping from doc_id to text (for LLM judge)
        """
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.llm_judge = llm_judge
        self.weight_policy = weight_policy
        self.top_k = top_k
        self._doc_map: Dict[str, str] = doc_map or {}
    
    def build_index(self, docs: List[Document]) -> None:
        """Build indices for both retrievers and store document texts."""
        _log.info(f"🚀 Building DAT Hybrid Index ({len(docs)} documentos)")
        
        # Store document texts for LLM judge
        for doc in docs:
            doc_text = (doc.title or "") + " " + (doc.text or "")
            self._doc_map[doc.doc_id] = doc_text
        
        # Build BM25 index
        with log_time(_log, "Build BM25 index"):
            self.bm25_retriever.build_index(docs)
        
        # Build Dense index
        with log_time(_log, "Build Dense index"):
            self.dense_retriever.build_index(docs)
        
        _log.info("✅ DAT Hybrid Index built successfully")
    
    def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieve documents using DAT hybrid approach.
        
        Args:
            queries: List of queries
            k: Number of results to return (default: 10)
        
        Returns:
            Dictionary mapping query_id to list of (doc_id, score) tuples
        """
        _log.info(f"🔍 DAT Hybrid Retrieval: {len(queries)} queries, top-{k}")
        
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
            
            # Step 2: Extract top-1
            bm25_top1_id, _ = bm25_topk[0]
            dense_top1_id, _ = dense_topk[0]
            
            # Get texts from doc_map (populated during build_index)
            bm25_top1_text = self._doc_map.get(bm25_top1_id, "")
            dense_top1_text = self._doc_map.get(dense_top1_id, "")
            
            # Fallback: try to get from BM25 retriever if not in doc_map
            if not bm25_top1_text and hasattr(self.bm25_retriever, 'doc_texts') and hasattr(self.bm25_retriever, 'doc_ids'):
                try:
                    idx = self.bm25_retriever.doc_ids.index(bm25_top1_id)
                    bm25_top1_text = self.bm25_retriever.doc_texts[idx]
                except (ValueError, IndexError):
                    pass
            
            if not bm25_top1_text or not dense_top1_text:
                _log.warning(f"Missing text for top-1 documents (bm25={bm25_top1_id}, dense={dense_top1_id}). Using empty strings.")
                bm25_top1_text = bm25_top1_text or ""
                dense_top1_text = dense_top1_text or ""
            
            # Step 3: Evaluate effectiveness via LLM judge (both together)
            try:
                e_dense, e_bm25 = self.llm_judge.evaluate_pair(
                    query=q.text,
                    dense_text=dense_top1_text,
                    bm25_text=bm25_top1_text,
                    query_id=q.query_id,
                    dense_doc_id=dense_top1_id,
                    bm25_doc_id=bm25_top1_id,
                )
                _log.debug(f"Query {q.query_id}: e_dense={e_dense}, e_bm25={e_bm25}")
            except Exception as e:
                _log.error(f"LLM judge failed for query {q.query_id}: {e}")
                # Fallback: use equal weights
                e_dense, e_bm25 = 3, 3
            
            # Step 4: Normalize judge scores: e^d = e_d/5, e^b = e_b/5
            # (This is done inside compute_alpha)
            
            # Step 5: Compute α(q)
            alpha = self.weight_policy.compute_alpha(e_dense, e_bm25)
            _log.debug(f"Query {q.query_id}: α(q)={alpha}")
            
            # Step 6: Normalize scores separately (min-max per method)
            bm25_normalized = normalize_scores_minmax(bm25_topk)
            dense_normalized = normalize_scores_minmax(dense_topk)
            
            # Step 7-9: Fuse using DATLinearFusion
            fusion_strategy = DATLinearFusion(alpha=alpha)
            fused_results = fusion_strategy.fuse(
                query=q.text,
                results_list=[
                    {q.query_id: bm25_normalized},
                    {q.query_id: dense_normalized},
                ],
            )
            
            # Step 10: Return top-k
            results[q.query_id] = fused_results[:k]
        
        _log.info(f"✅ DAT Hybrid Retrieval completed: {len(results)} queries")
        return results

