"""
Fusion strategies for combining multiple retrieval results.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from ..core.interfaces import AbstractFusionStrategy


class WeightedCosineFusion(AbstractFusionStrategy):
    """
    Weighted cosine fusion strategy.
    Combines results by weighted sum of cosine similarities.
    """
    
    def fuse(
        self,
        query: str,
        results_list: List[Dict[str, List[Tuple[str, float]]]],
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """Fuse multiple retrieval results using weighted cosine."""
        if not results_list:
            return []
        
        if weights is None:
            weights = [1.0 / len(results_list)] * len(results_list)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(results_list)] * len(results_list)
        
        # Combine scores
        doc_scores: Dict[str, float] = {}
        for result_dict, weight in zip(results_list, weights):
            # Assume all results are for the same query (first query_id)
            query_id = next(iter(result_dict.keys())) if result_dict else None
            if query_id:
                for doc_id, score in result_dict[query_id]:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + weight * score
        
        # Sort by score
        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)


class ReciprocalRankFusion(AbstractFusionStrategy):
    """
    Reciprocal Rank Fusion (RRF) strategy.
    Combines results using RRF formula: score = sum(1 / (k + rank))
    """
    
    def __init__(self, k: int = 60):
        """
        Args:
            k: RRF constant (typically 60)
        """
        self.k = k
    
    def fuse(
        self,
        query: str,
        results_list: List[Dict[str, List[Tuple[str, float]]]],
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """Fuse multiple retrieval results using RRF."""
        if not results_list:
            return []
        
        if weights is None:
            weights = [1.0] * len(results_list)
        
        # Build rank map for each result set
        doc_ranks: Dict[str, List[int]] = {}
        for result_dict, weight in zip(results_list, weights):
            query_id = next(iter(result_dict.keys())) if result_dict else None
            if query_id:
                for rank, (doc_id, _) in enumerate(result_dict[query_id], start=1):
                    if doc_id not in doc_ranks:
                        doc_ranks[doc_id] = []
                    doc_ranks[doc_id].append(rank * weight)  # Weighted rank
        
        # Compute RRF scores
        doc_scores: Dict[str, float] = {}
        for doc_id, ranks in doc_ranks.items():
            # RRF: sum of 1 / (k + rank) for each rank
            score = sum(1.0 / (self.k + rank) for rank in ranks)
            doc_scores[doc_id] = score
        
        # Sort by score
        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)


class DATLinearFusion(AbstractFusionStrategy):
    """
    DAT (Dynamic Alpha Tuning) Linear Fusion strategy.
    Implements: R(q,d) = α(q) * S̃_dense(q,d) + (1-α(q)) * S̃_BM25(q,d)
    
    Process:
    1. Normalize scores BM25@K separately (min-max)
    2. Normalize scores Dense@K separately (min-max)
    3. Union doc_ids (BM25@K ∪ Dense@K)
    4. For each doc in union: assign score = 0 if not in a method
    5. Compute R(q,d) for each doc
    6. Sort by R(q,d) with stable tie-breaking
    """
    
    def __init__(self, alpha: float):
        """
        Args:
            alpha: Alpha value for fusion (0.0 to 1.0)
        """
        self.alpha = alpha
    
    def fuse(
        self,
        query: str,
        results_list: List[Dict[str, List[Tuple[str, float]]]],
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Fuse BM25 and Dense results using linear combination with alpha.
        
        Args:
            query: Query text (not used, but required by interface)
            results_list: List of two result dicts [BM25_results, Dense_results]
            weights: Optional weights (ignored, uses self.alpha)
        
        Returns:
            List of (doc_id, R(q,d)) tuples sorted by R(q,d) desc
        """
        if not results_list or len(results_list) != 2:
            raise ValueError("DATLinearFusion requires exactly 2 result sets (BM25 and Dense)")
        
        # Extract results (assume first is BM25, second is Dense)
        bm25_results = results_list[0]
        dense_results = results_list[1]
        
        # Get query_id (should be the same in both)
        query_id = next(iter(bm25_results.keys())) if bm25_results else None
        if not query_id:
            return []
        
        # Get scores as dicts for easy lookup
        bm25_scores = dict(bm25_results.get(query_id, []))
        dense_scores = dict(dense_results.get(query_id, []))
        
        # Union of doc_ids
        all_doc_ids = set(bm25_scores.keys()) | set(dense_scores.keys())
        
        # Compute R(q,d) for each doc
        combined_scores = {}
        for doc_id in all_doc_ids:
            s_bm25 = bm25_scores.get(doc_id, 0.0)
            s_dense = dense_scores.get(doc_id, 0.0)
            
            # Linear combination: R(q,d) = α * S̃_dense + (1-α) * S̃_BM25
            r_score = self.alpha * s_dense + (1.0 - self.alpha) * s_bm25
            combined_scores[doc_id] = (r_score, s_dense, s_bm25)
        
        # Sort by: 1) R(q,d) desc, 2) max(S̃_dense, S̃_BM25) desc, 3) doc_id (lexicographic)
        sorted_docs = sorted(
            combined_scores.items(),
            key=lambda x: (
                -x[1][0],  # R(q,d) descending (negative for desc)
                -max(x[1][1], x[1][2]),  # max(S̃_dense, S̃_BM25) descending
                x[0]  # doc_id ascending (lexicographic, for stability)
            )
        )
        
        # Return only (doc_id, R(q,d))
        return [(doc_id, r_score) for doc_id, (r_score, _, _) in sorted_docs]


# Registry of fusion strategies
FUSION_STRATEGIES: Dict[str, type[AbstractFusionStrategy]] = {
    "weighted_cosine": WeightedCosineFusion,
    "reciprocal_rank": ReciprocalRankFusion,
    "dat_linear": DATLinearFusion,
}


def create_fusion_strategy(strategy_name: str, **kwargs) -> AbstractFusionStrategy:
    """Create a fusion strategy by name.
    
    Args:
        strategy_name: Name of the fusion strategy
        **kwargs: Strategy-specific parameters
        
    Returns:
        Fusion strategy instance
    """
    strategy_name = strategy_name.lower()
    if strategy_name not in FUSION_STRATEGIES:
        raise ValueError(
            f"Unknown fusion strategy: {strategy_name}. "
            f"Available: {list(FUSION_STRATEGIES.keys())}"
        )
    
    strategy_class = FUSION_STRATEGIES[strategy_name]
    return strategy_class(**kwargs)