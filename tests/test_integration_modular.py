"""
Integration tests for the modular system.
Tests that components work together correctly.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.datasets.schema import Document, Query
from src.vectorizers.factory import create_vectorizer
from src.indexes.factory import create_index
from src.fusion.factory import create_weight_policy, create_reranker
from src.retrievers.factory import create_retriever
from src.eval.evaluator import evaluate_predictions
from src.eval.formatters import get_formatter
from src.config.loader import load_config


class TestModularIntegration:
    """Test that modular components work together."""
    
    def test_vectorizer_to_index_flow(self):
        """Test that vectorizer can be used to build index."""
        vec_config = {"type": "tfidf", "tfidf": {"dim": 100, "min_df": 1}}
        vec = create_vectorizer(vec_config)
        vec.fit_corpus(["doc1 text", "doc2 text", "doc3 text"])
        
        idx_config = {"type": "faiss", "metric": "ip"}
        index = create_index(idx_config, vec)
        
        # Build index
        doc_pairs = [("d1", "doc1 text"), ("d2", "doc2 text"), ("d3", "doc3 text")]
        index.build(doc_pairs)
        
        # Search
        query_vec = vec.concat(vec.encode_text("doc1", is_query=True))
        results = index.search(query_vec, topk=2)
        
        assert len(results) == 2
        assert results[0][0] in ["d1", "d2", "d3"]
    
    def test_retriever_build_and_retrieve(self):
        """Test that retriever can build index and retrieve."""
        config = {
            "type": "hybrid",
            "vectorizer": {
                "type": "tfidf",
                "tfidf": {"dim": 50, "min_df": 1}
            },
            "fusion": {
                "policy": "static",
                "weights": [1.0]
            },
            "reranker": {
                "type": "none"
            },
            "index": {
                "type": "faiss"
            }
        }
        
        retriever = create_retriever(config)
        
        # Build index
        docs = [
            Document("d1", "title1", "This is document one", None),
            Document("d2", "title2", "This is document two", None),
            Document("d3", "title3", "This is document three", None),
        ]
        retriever.build_index(docs)
        
        # Retrieve
        queries = [
            Query("q1", "document one"),
            Query("q2", "document two"),
        ]
        results = retriever.retrieve(queries, k=2)
        
        assert "q1" in results
        assert "q2" in results
        assert len(results["q1"]) <= 2
        assert len(results["q2"]) <= 2
    
    def test_evaluation_with_custom_metrics(self):
        """Test evaluation with custom metric selection."""
        preds = {
            "q1": [("d1", 0.9), ("d2", 0.8), ("d3", 0.7)],
            "q2": [("d2", 0.9), ("d1", 0.8)],
        }
        
        qrels = pd.DataFrame([
            {"query_id": "q1", "doc_id": "d1", "score": 1, "split": "test"},
            {"query_id": "q1", "doc_id": "d2", "score": 1, "split": "test"},
            {"query_id": "q2", "doc_id": "d2", "score": 1, "split": "test"},
        ])
        
        # Evaluate with only MRR and nDCG
        results = evaluate_predictions(preds, qrels, ks=[1, 2], metrics=["MRR", "nDCG"])
        
        assert "MRR" in results.columns
        assert "nDCG" in results.columns
        assert "MAP" not in results.columns  # Not requested
        assert len(results) == 2  # Two k values
    
    def test_formatters_with_results(self):
        """Test that formatters work with evaluation results."""
        results_df = pd.DataFrame([
            {"k": 1, "MRR": 0.5, "nDCG": 0.6},
            {"k": 3, "MRR": 0.7, "nDCG": 0.8},
        ])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # CSV
            csv_fmt = get_formatter("csv")
            csv_path = Path(tmpdir) / "results.csv"
            csv_fmt.format(results_df, output_path=str(csv_path))
            assert csv_path.exists()
            
            # JSON
            json_fmt = get_formatter("json")
            json_path = Path(tmpdir) / "results.json"
            json_fmt.format(results_df, output_path=str(json_path))
            assert json_path.exists()
            
            # JSONL
            jsonl_fmt = get_formatter("jsonl")
            jsonl_path = Path(tmpdir) / "results.jsonl"
            jsonl_fmt.format(results_df, output_path=str(jsonl_path))
            assert jsonl_path.exists()
    
    def test_config_loading_and_validation(self):
        """Test that config files are loaded and validated correctly."""
        # Create a minimal valid config
        config_dict = {
            "dataset": {
                "name": "scifact",
                "root": "./data/scifact/processed/beir"
            },
            "retrievers": [
                {
                    "type": "tfidf",
                    "tfidf": {"dim": 100}
                }
            ],
            "metrics": ["MRR", "nDCG"],
            "ks": [1, 3],
            "output_formats": ["csv"]
        }
        
        from src.config.schema import ExperimentConfig
        config = ExperimentConfig(**config_dict)
        
        assert config.dataset.name == "scifact"
        assert len(config.retrievers) == 1
        assert config.retrievers[0].type == "tfidf"
        assert "MRR" in config.metrics
        assert "nDCG" in config.metrics

