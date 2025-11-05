"""
Tests for metric classes implementing AbstractMetric interface.
"""

import pytest
from src.eval.metrics import (
    MRRMetric,
    NDCGMetric,
    MAPMetric,
    RecallMetric,
    PrecisionMetric,
    get_metric,
    METRICS_REGISTRY,
)


class TestMRRMetric:
    def test_mrr_perfect_first(self):
        """Test MRR when first result is relevant."""
        metric = MRRMetric()
        ranked = ["d1", "d2", "d3"]
        gold = {"d1": 1.0, "d2": 1.0}
        
        assert metric.compute(ranked, gold, k=3) == 1.0
    
    def test_mrr_second_relevant(self):
        """Test MRR when second result is relevant."""
        metric = MRRMetric()
        ranked = ["d1", "d2", "d3"]
        gold = {"d2": 1.0, "d3": 1.0}
        
        assert metric.compute(ranked, gold, k=3) == pytest.approx(0.5)
    
    def test_mrr_no_relevant(self):
        """Test MRR when no relevant documents."""
        metric = MRRMetric()
        ranked = ["d1", "d2", "d3"]
        gold = {"d4": 1.0}
        
        assert metric.compute(ranked, gold, k=3) == 0.0
    
    def test_mrr_k_cutoff(self):
        """Test MRR respects k cutoff."""
        metric = MRRMetric()
        ranked = ["d1", "d2", "d3", "d4"]
        gold = {"d3": 1.0}
        
        assert metric.compute(ranked, gold, k=2) == 0.0
        assert metric.compute(ranked, gold, k=3) == pytest.approx(1.0 / 3)


class TestNDCGMetric:
    def test_ndcg_perfect(self):
        """Test nDCG with perfect ranking."""
        metric = NDCGMetric()
        ranked = ["d1", "d2", "d3"]
        gold = {"d1": 2.0, "d2": 1.0, "d3": 0.0}
        
        result = metric.compute(ranked, gold, k=3)
        assert result == pytest.approx(1.0, abs=1e-6)
    
    def test_ndcg_imperfect(self):
        """Test nDCG with imperfect ranking."""
        metric = NDCGMetric()
        ranked = ["d2", "d1", "d3"]  # Wrong order
        gold = {"d1": 2.0, "d2": 1.0, "d3": 0.0}
        
        result = metric.compute(ranked, gold, k=3)
        assert 0.0 < result < 1.0
    
    def test_ndcg_no_relevant(self):
        """Test nDCG when no relevant documents."""
        metric = NDCGMetric()
        ranked = ["d1", "d2", "d3"]
        gold = {"d1": 0.0, "d2": 0.0, "d3": 0.0}
        
        assert metric.compute(ranked, gold, k=3) == 0.0


class TestMAPMetric:
    def test_map_perfect(self):
        """Test MAP with perfect ranking."""
        metric = MAPMetric()
        ranked = ["d1", "d2", "d3"]
        gold = {"d1": 1.0, "d2": 1.0}
        
        # Both relevant at positions 1 and 2: (1/1 + 2/2) / 2 = 1.0
        result = metric.compute(ranked, gold, k=3)
        assert result == pytest.approx(1.0)
    
    def test_map_partial(self):
        """Test MAP with partial matches."""
        metric = MAPMetric()
        ranked = ["d1", "d2", "d3", "d4"]
        gold = {"d2": 1.0, "d4": 1.0}
        
        # Relevant at positions 2 and 4: (1/2 + 2/4) / 2 = 0.5
        result = metric.compute(ranked, gold, k=4)
        assert result == pytest.approx(0.5)
    
    def test_map_no_relevant(self):
        """Test MAP when no relevant documents."""
        metric = MAPMetric()
        ranked = ["d1", "d2", "d3"]
        gold = {}
        
        assert metric.compute(ranked, gold, k=3) == 0.0


class TestRecallMetric:
    def test_recall_perfect(self):
        """Test Recall when all relevant are retrieved."""
        metric = RecallMetric()
        ranked = ["d1", "d2", "d3"]
        gold = {"d1": 1.0, "d2": 1.0}
        
        result = metric.compute(ranked, gold, k=3)
        assert result == 1.0
    
    def test_recall_partial(self):
        """Test Recall with partial retrieval."""
        metric = RecallMetric()
        ranked = ["d1", "d2", "d3"]
        gold = {"d1": 1.0, "d2": 1.0, "d4": 1.0}
        
        # 2 out of 3 relevant retrieved
        result = metric.compute(ranked, gold, k=3)
        assert result == pytest.approx(2.0 / 3.0)
    
    def test_recall_k_cutoff(self):
        """Test Recall respects k cutoff."""
        metric = RecallMetric()
        ranked = ["d1", "d2", "d3", "d4"]
        gold = {"d1": 1.0, "d3": 1.0, "d4": 1.0}
        
        # At k=2, only d1 retrieved: 1/3
        assert metric.compute(ranked, gold, k=2) == pytest.approx(1.0 / 3.0)
        # At k=3, d1 and d3 retrieved: 2/3
        assert metric.compute(ranked, gold, k=3) == pytest.approx(2.0 / 3.0)


class TestPrecisionMetric:
    def test_precision_perfect(self):
        """Test Precision when all retrieved are relevant."""
        metric = PrecisionMetric()
        ranked = ["d1", "d2", "d3"]
        gold = {"d1": 1.0, "d2": 1.0, "d3": 1.0}
        
        result = metric.compute(ranked, gold, k=3)
        assert result == 1.0
    
    def test_precision_partial(self):
        """Test Precision with partial relevance."""
        metric = PrecisionMetric()
        ranked = ["d1", "d2", "d3"]
        gold = {"d1": 1.0, "d2": 1.0}
        
        # 2 relevant out of 3 retrieved
        result = metric.compute(ranked, gold, k=3)
        assert result == pytest.approx(2.0 / 3.0)
    
    def test_precision_k_cutoff(self):
        """Test Precision respects k cutoff."""
        metric = PrecisionMetric()
        ranked = ["d1", "d2", "d3", "d4"]
        gold = {"d1": 1.0, "d3": 1.0}
        
        # At k=2, only d1 is relevant: 1/2
        assert metric.compute(ranked, gold, k=2) == pytest.approx(0.5)
        # At k=3, d1 and d3 are relevant: 2/3
        assert metric.compute(ranked, gold, k=3) == pytest.approx(2.0 / 3.0)
    
    def test_precision_k_zero(self):
        """Test Precision with k=0."""
        metric = PrecisionMetric()
        ranked = ["d1", "d2"]
        gold = {"d1": 1.0}
        
        assert metric.compute(ranked, gold, k=0) == 0.0


class TestMetricRegistry:
    def test_get_metric_exact_match(self):
        """Test getting metric by exact name."""
        metric = get_metric("nDCG")
        assert isinstance(metric, NDCGMetric)
    
    def test_get_metric_case_insensitive(self):
        """Test getting metric case-insensitively."""
        metric1 = get_metric("mrr")
        metric2 = get_metric("MRR")
        assert isinstance(metric1, MRRMetric)
        assert isinstance(metric2, MRRMetric)
        assert metric1 is metric2
    
    def test_get_metric_invalid(self):
        """Test error for invalid metric name."""
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("invalid_metric")
    
    def test_all_metrics_in_registry(self):
        """Test that all metric classes are in registry."""
        expected_metrics = {"MRR", "nDCG", "MAP", "Recall", "Precision"}
        assert set(METRICS_REGISTRY.keys()) == expected_metrics

