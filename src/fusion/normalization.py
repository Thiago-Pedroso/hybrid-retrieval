"""
Normalization utilities for scores per query.
Implements min-max normalization with special handling for edge cases.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from ..utils.logging import get_logger

_log = get_logger("fusion.normalization")


def normalize_scores_minmax(
    scores: List[Tuple[str, float]],
    qrels: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float]]:
    """
    Normalize scores using min-max normalization per query.
    
    Normalization formula: (score - min) / (max - min)
    
    Special case when max == min:
    - Literal paper version: all scores = 0.0
    - Optional with qrels: if any candidate is relevant, all = 1.0; else all = 0.0
    
    Args:
        scores: List of (doc_id, score) tuples to normalize
        qrels: Optional dict mapping doc_id to relevance score (for special case handling)
    
    Returns:
        List of (doc_id, normalized_score) tuples
    """
    if not scores:
        return []
    
    # Extract scores
    score_values = [score for _, score in scores]
    
    min_score = min(score_values)
    max_score = max(score_values)
    
    # Special case: all scores are equal
    if max_score == min_score:
        # Literal paper version: all = 0.0
        if qrels is None:
            _log.debug(f"All scores equal ({max_score}), setting all to 0.0")
            return [(doc_id, 0.0) for doc_id, _ in scores]
        
        # Optional version with qrels: check if any is relevant
        has_relevant = any(
            doc_id in qrels and qrels[doc_id] > 0
            for doc_id, _ in scores
        )
        if has_relevant:
            _log.debug(f"All scores equal ({max_score}), but has relevant docs, setting all to 1.0")
            return [(doc_id, 1.0) for doc_id, _ in scores]
        else:
            _log.debug(f"All scores equal ({max_score}), no relevant docs, setting all to 0.0")
            return [(doc_id, 0.0) for doc_id, _ in scores]
    
    # Normal min-max normalization
    score_range = max_score - min_score
    if score_range == 0:
        # Should not happen (already handled above), but just in case
        return [(doc_id, 0.0) for doc_id, _ in scores]
    
    normalized = [
        (doc_id, (score - min_score) / score_range)
        for doc_id, score in scores
    ]
    
    return normalized
