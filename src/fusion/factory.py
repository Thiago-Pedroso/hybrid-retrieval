"""
Factory for creating fusion components (policies, rerankers) from configuration.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
from ..core.interfaces import AbstractWeightPolicy, AbstractReranker
from .weighting import StaticPolicy, HeuristicLLMPolicy, DATWeightPolicy
from .reranker import TriModalReranker, BiModalReranker
from .strategies import create_fusion_strategy, AbstractFusionStrategy
from .llm_judge import LLMJudge


def create_weight_policy(config: Dict[str, Any]) -> AbstractWeightPolicy:
    """Create a weight policy from configuration.
    
    Args:
        config: Configuration with 'policy' key and optional 'weights'
        
    Returns:
        Weight policy instance
    """
    policy_type = config.get("policy", "heuristic").lower()
    
    if policy_type == "static":
        weights = config.get("weights", [0.5, 0.3, 0.2])
        if len(weights) >= 3:
            return StaticPolicy(ws=weights[0], wt=weights[1], wg=weights[2])
        elif len(weights) == 2:
            return StaticPolicy(ws=weights[0], wt=weights[1], wg=0.0)
        else:
            return StaticPolicy()
    
    elif policy_type == "heuristic":
        return HeuristicLLMPolicy()
    
    else:
        raise ValueError(f"Unknown weight policy: {policy_type}. Available: static, heuristic")


def create_reranker(
    reranker_type: str,
    vectorizer: Any,  # AbstractVectorizer, but avoiding circular import
) -> Optional[AbstractReranker]:
    """Create a reranker from type and vectorizer.
    
    Args:
        reranker_type: Type of reranker ("tri_modal", "bi_modal", "none")
        vectorizer: Vectorizer instance (must match reranker type)
        
    Returns:
        Reranker instance or None if type is "none"
    """
    if reranker_type is None or reranker_type.lower() == "none":
        return None
    
    reranker_type = reranker_type.lower()
    
    if reranker_type == "tri_modal":
        from ..vectorizers.tri_modal_vectorizer import TriModalVectorizer
        if not isinstance(vectorizer, TriModalVectorizer):
            raise TypeError(f"TriModalReranker requires TriModalVectorizer, got {type(vectorizer)}")
        return TriModalReranker(vectorizer)
    
    elif reranker_type == "bi_modal":
        from ..vectorizers.bi_modal_vectorizer import BiModalVectorizer
        if not isinstance(vectorizer, BiModalVectorizer):
            raise TypeError(f"BiModalReranker requires BiModalVectorizer, got {type(vectorizer)}")
        return BiModalReranker(vectorizer)
    
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}. Available: tri_modal, bi_modal, none")


def create_fusion_strategy_from_config(config: Dict[str, Any]) -> AbstractFusionStrategy:
    """Create a fusion strategy from configuration.
    
    Args:
        config: Configuration with 'strategy' key and strategy-specific params
        
    Returns:
        Fusion strategy instance
    """
    strategy_name = config.get("strategy", "weighted_cosine").lower()
    kwargs = {}
    
    if strategy_name == "reciprocal_rank":
        kwargs["k"] = config.get("k", 60)
    
    return create_fusion_strategy(strategy_name, **kwargs)


def create_llm_judge(config: Dict[str, Any]) -> LLMJudge:
    """Create an LLM judge from configuration.
    
    Args:
        config: Configuration with LLM judge parameters
        
    Returns:
        LLMJudge instance
    """
    return LLMJudge(
        model=config.get("model", "gpt-4o-mini"),
        temperature=config.get("temperature", 0.0),
        max_tokens=config.get("max_tokens", 10),
        prompt_template=config.get("prompt_template"),
        cache_dir=config.get("cache_dir"),
        timeout=config.get("timeout", 30),
        max_retries=config.get("max_retries", 3),
        api_key=config.get("api_key"),
        max_text_tokens=config.get("max_text_tokens", 2000),
        rate_limit_tier=config.get("rate_limit_tier", 2),
        rate_limit_safety_margin=config.get("rate_limit_safety_margin", 0.1),
    )


def create_dat_weight_policy() -> DATWeightPolicy:
    """Create a DAT weight policy.
    
    Returns:
        DATWeightPolicy instance
    """
    return DATWeightPolicy()