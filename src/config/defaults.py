"""
Default configurations for components.
"""

from typing import Dict, Any
from .schema import ExperimentConfig, RetrieverConfig, DatasetConfig, VectorizerConfig


def get_default_config() -> Dict[str, Any]:
    """Get default configuration dictionary."""
    return {
        "experiment": {
            "name": "default_experiment",
            "output_dir": "./outputs/experiments/default",
        },
        "dataset": {
            "name": "scifact",
            "root": "./data/scifact/processed/beir",
            "split_preference": ["test", "dev", "validation", "train"],
        },
        "retrievers": [
            {
                "name": "hybrid_default",
                "type": "hybrid",
                "vectorizer": {
                    "type": "tri_modal",
                    "semantic": {
                        "model": "sentence-transformers/all-MiniLM-L6-v2",
                        "device": None,
                    },
                    "tfidf": {
                        "dim": 1000,
                        "min_df": 2,
                        "backend": "sklearn",
                    },
                    "graph": {
                        "model": "BAAI/bge-large-en-v1.5",
                        "ner_backend": "scispacy",
                    },
                },
                "fusion": {
                    "strategy": "weighted_cosine",
                    "policy": "heuristic",
                },
                "reranker": {
                    "type": "tri_modal",
                    "topk_first": 150,
                },
                "index": {
                    "type": "faiss",
                    "factory": None,
                    "metric": "ip",
                },
            }
        ],
        "metrics": ["nDCG", "MRR", "MAP", "Recall", "Precision"],
        "ks": [1, 3, 5, 10],
        "output_formats": ["csv"],
        "output_dir": "./outputs/experiments/default",
    }


def get_default_retriever_config() -> Dict[str, Any]:
    """Get default retriever configuration."""
    return {
        "type": "hybrid",
        "vectorizer": {
            "type": "tri_modal",
            "semantic": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "tfidf": {
                "dim": 1000,
            },
            "graph": {
                "model": "BAAI/bge-large-en-v1.5",
            },
        },
        "fusion": {
            "strategy": "weighted_cosine",
            "policy": "heuristic",
        },
        "reranker": {
            "type": "tri_modal",
            "topk_first": 150,
        },
    }

