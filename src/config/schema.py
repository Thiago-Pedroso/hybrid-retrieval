"""
Pydantic schemas for configuration validation.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator


class SemanticConfig(BaseModel):
    """Configuration for semantic encoder."""
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None
    query_prefix: str = ""
    doc_prefix: str = ""


class TFIDFConfig(BaseModel):
    """Configuration for TF-IDF encoder."""
    dim: Optional[int] = 1000
    min_df: int = 2
    backend: Literal["sklearn", "pyserini"] = "sklearn"
    scale_multiplier: float = 1.25


class GraphConfig(BaseModel):
    """Configuration for graph/entity encoder."""
    model: str = "BAAI/bge-large-en-v1.5"
    ner_backend: Literal["scispacy", "spacy", "none"] = "scispacy"
    ner_model: Optional[str] = None
    ner_use_noun_chunks: bool = True
    ner_batch_size: int = 128
    ner_n_process: int = 4
    ner_allowed_labels: Optional[List[str]] = None
    entity_min_df: Optional[int] = None
    entity_max_entities_per_text: Optional[int] = 128
    entity_artifact_dir: Optional[str] = None
    entity_force_rebuild: bool = False


class VectorizerConfig(BaseModel):
    """Configuration for vectorizer."""
    type: Literal["dense", "tfidf", "bi_modal", "tri_modal", "graph"] = "tri_modal"
    semantic: Optional[SemanticConfig] = None
    tfidf: Optional[TFIDFConfig] = None
    graph: Optional[GraphConfig] = None


class FusionConfig(BaseModel):
    """Configuration for fusion strategy."""
    strategy: Literal["weighted_cosine", "reciprocal_rank", "learned"] = "weighted_cosine"
    policy: Literal["static", "heuristic"] = "heuristic"
    weights: Optional[List[float]] = None  # For static policy


class RerankerConfig(BaseModel):
    """Configuration for reranker."""
    type: Literal["tri_modal", "bi_modal", "none"] = "tri_modal"
    topk_first: int = 150


class IndexConfig(BaseModel):
    """Configuration for search index."""
    type: Literal["faiss", "numpy"] = "faiss"
    factory: Optional[str] = None  # e.g., "OPQ64,IVF4096,PQ64x8" or null for FlatIP
    metric: Literal["ip", "l2"] = "ip"
    nprobe: Optional[int] = None
    train_size: int = 0  # 0 = auto
    artifact_dir: Optional[str] = None
    index_name: str = "index"


class RetrieverConfig(BaseModel):
    """Configuration for a retriever."""
    name: Optional[str] = None
    type: Literal["hybrid", "dense", "tfidf", "graph", "bm25"] = "hybrid"
    vectorizer: Optional[VectorizerConfig] = None
    fusion: Optional[FusionConfig] = None
    reranker: Optional[RerankerConfig] = None
    index: Optional[IndexConfig] = None
    # Additional retriever-specific config (for backward compatibility with legacy params)
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    # Legacy parameters (for backward compatibility)
    semantic: Optional[Dict[str, Any]] = None
    tfidf: Optional[Dict[str, Any]] = None
    graph: Optional[Dict[str, Any]] = None


class DatasetConfig(BaseModel):
    """Configuration for dataset."""
    name: str
    root: Optional[str] = None
    split_preference: List[str] = Field(default_factory=lambda: ["test", "dev", "validation", "train"])
    qrels_path: Optional[str] = None


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    experiment: Dict[str, Any] = Field(default_factory=dict)
    # Support both single dataset (backward compatibility) and multiple datasets
    dataset: Optional[DatasetConfig] = None
    datasets: Optional[List[DatasetConfig]] = None
    retrievers: List[RetrieverConfig]
    metrics: List[str] = Field(default_factory=lambda: ["nDCG", "MRR", "MAP", "Recall", "Precision"])
    ks: List[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    output_formats: List[str] = Field(default_factory=lambda: ["csv"])
    output_dir: Optional[str] = None

    @field_validator("retrievers")
    @classmethod
    def validate_retrievers(cls, v: List[RetrieverConfig]) -> List[RetrieverConfig]:
        if not v:
            raise ValueError("At least one retriever must be specified")
        return v

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: List[str]) -> List[str]:
        valid_metrics = {"nDCG", "MRR", "MAP", "Recall", "Precision"}
        invalid = set(v) - valid_metrics
        if invalid:
            raise ValueError(f"Invalid metrics: {invalid}. Valid: {valid_metrics}")
        return v
    
    @model_validator(mode="after")
    def validate_dataset_config(self):
        """Ensure at least one dataset is specified."""
        if self.dataset is None and (self.datasets is None or len(self.datasets) == 0):
            raise ValueError("Either 'dataset' or 'datasets' must be specified")
        return self
    
    def get_datasets(self) -> List[DatasetConfig]:
        """Get list of datasets to process."""
        if self.datasets is not None:
            return self.datasets
        elif self.dataset is not None:
            return [self.dataset]
        else:
            raise ValueError("No datasets specified")

