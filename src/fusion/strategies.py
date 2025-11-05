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


# Registry of fusion strategies
FUSION_STRATEGIES: Dict[str, type[AbstractFusionStrategy]] = {
    "weighted_cosine": WeightedCosineFusion,
    "reciprocal_rank": ReciprocalRankFusion,
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