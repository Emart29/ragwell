"""Tests for evaluation metrics."""
import pytest
import math

from evaluation.eval_metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    MetricsResult
)
from app.retrieval.types import SearchResult


class TestMRR:
    """Tests for Mean Reciprocal Rank."""
    
    def test_mrr_perfect(self):
        """Test MRR when relevant result is always first."""
        results = [
            [SearchResult(chunk_id='a', text='text', score=1.0, document_id='d1', filename='f1')],
            [SearchResult(chunk_id='b', text='text', score=1.0, document_id='d1', filename='f1')]
        ]
        relevant = [['a'], ['b']]
        
        mrr = mean_reciprocal_rank(results, relevant)
        assert mrr == 1.0
    
    def test_mrr_second_position(self):
        """Test MRR when relevant result is second."""
        results = [
            [
                SearchResult(chunk_id='x', text='text', score=1.0, document_id='d1', filename='f1'),
                SearchResult(chunk_id='a', text='text', score=0.9, document_id='d1', filename='f1')
            ]
        ]
        relevant = [['a']]
        
        mrr = mean_reciprocal_rank(results, relevant)
        assert mrr == 0.5
    
    def test_mrr_no_relevant(self):
        """Test MRR when no relevant results."""
        results = [[SearchResult(chunk_id='x', text='text', score=1.0, document_id='d1', filename='f1')]]
        relevant = [['a']]
        
        mrr = mean_reciprocal_rank(results, relevant)
        assert mrr == 0.0
    
    def test_mrr_mixed(self):
        """Test MRR with mixed rankings."""
        results = [
            [SearchResult(chunk_id='a', text='text', score=1.0, document_id='d1', filename='f1')],  # First
            [
                SearchResult(chunk_id='x', text='text', score=1.0, document_id='d1', filename='f1'),
                SearchResult(chunk_id='b', text='text', score=0.9, document_id='d1', filename='f1')  # Second
            ]
        ]
        relevant = [['a'], ['b']]
        
        mrr = mean_reciprocal_rank(results, relevant)
        assert mrr == (1.0 + 0.5) / 2


class TestRecallAtK:
    """Tests for Recall@K."""
    
    def test_recall_at_1_hit(self):
        """Test Recall@1 when relevant doc is first."""
        results = [[SearchResult(chunk_id='a', text='text', score=1.0, document_id='d1', filename='f1')]]
        relevant = [['a']]
        
        recall = recall_at_k(results, relevant, 1)
        assert recall == 1.0
    
    def test_recall_at_1_miss(self):
        """Test Recall@1 when relevant doc is not first."""
        results = [
            [
                SearchResult(chunk_id='x', text='text', score=1.0, document_id='d1', filename='f1'),
                SearchResult(chunk_id='a', text='text', score=0.9, document_id='d1', filename='f1')
            ]
        ]
        relevant = [['a']]
        
        recall = recall_at_k(results, relevant, 1)
        assert recall == 0.0
    
    def test_recall_at_5(self):
        """Test Recall@5."""
        results = [
            [
                SearchResult(chunk_id='x', text='text', score=1.0, document_id='d1', filename='f1'),
                SearchResult(chunk_id='a', text='text', score=0.9, document_id='d1', filename='f1')
            ]
        ]
        relevant = [['a']]
        
        recall = recall_at_k(results, relevant, 5)
        assert recall == 1.0


class TestDCG:
    """Tests for DCG."""
    
    def test_dcg_at_1(self):
        """Test DCG@1."""
        relevances = [1.0, 0.5, 0.0]
        dcg = dcg_at_k(relevances, 1)
        assert dcg == 1.0
    
    def test_dcg_calculation(self):
        """Test DCG calculation."""
        # DCG@3 = 1.0 + 0.5/log2(3) + 0.0/log2(4)
        relevances = [1.0, 0.5, 0.0]
        dcg = dcg_at_k(relevances, 3)
        expected = 1.0 + 0.5 / math.log2(3)
        assert abs(dcg - expected) < 0.001


class TestNDCG:
    """Tests for nDCG."""
    
    def test_ndcg_perfect(self):
        """Test nDCG with perfect ranking."""
        results = [[SearchResult(chunk_id='a', text='text', score=1.0, document_id='d1', filename='f1')]]
        relevant = [['a']]
        
        ndcg = ndcg_at_k(results, relevant, 5)
        assert ndcg == 1.0
    
    def test_ndcg_zero(self):
        """Test nDCG with no relevant results."""
        results = [[SearchResult(chunk_id='x', text='text', score=1.0, document_id='d1', filename='f1')]]
        relevant = [['a']]
        
        ndcg = ndcg_at_k(results, relevant, 5)
        assert ndcg == 0.0


class TestPrecisionAtK:
    """Tests for Precision@K."""
    
    def test_precision_at_5(self):
        """Test Precision@5."""
        results = [
            [
                SearchResult(chunk_id='a', text='text', score=1.0, document_id='d1', filename='f1'),
                SearchResult(chunk_id='b', text='text', score=0.9, document_id='d1', filename='f1'),
                SearchResult(chunk_id='x', text='text', score=0.8, document_id='d1', filename='f1'),
                SearchResult(chunk_id='y', text='text', score=0.7, document_id='d1', filename='f1'),
                SearchResult(chunk_id='z', text='text', score=0.6, document_id='d1', filename='f1')
            ]
        ]
        relevant = [['a', 'b']]
        
        precision = precision_at_k(results, relevant, 5)
        assert precision == 2/5  # 2 relevant out of 5


class TestMetricsResult:
    """Tests for MetricsResult class."""
    
    def test_metrics_result_to_dict(self):
        """Test conversion to dictionary."""
        metrics = MetricsResult(
            mrr=0.75,
            recall_1=0.5,
            recall_3=0.75,
            recall_5=0.9,
            recall_10=1.0,
            ndcg_5=0.8,
            ndcg_10=0.85,
            precision_5=0.6,
            latency_ms=100.5
        )
        
        d = metrics.to_dict()
        assert d['mrr'] == 0.75
        assert d['recall_1'] == 0.5
        assert d['latency_ms'] == 100.5
    
    def test_metrics_result_compute(self):
        """Test computing metrics from results."""
        results = [
            [SearchResult(chunk_id='a', text='text', score=1.0, document_id='d1', filename='f1')],
            [
                SearchResult(chunk_id='x', text='text', score=1.0, document_id='d1', filename='f1'),
                SearchResult(chunk_id='b', text='text', score=0.9, document_id='d1', filename='f1')
            ]
        ]
        relevant = [['a'], ['b']]
        latencies = [100.0, 150.0]
        
        metrics = MetricsResult.compute(results, relevant, latencies)
        
        assert metrics.mrr == (1.0 + 0.5) / 2
        assert metrics.recall_1 == 0.5  # Only first query has relevant at position 1
        assert metrics.latency_ms == 125.0  # Average


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
