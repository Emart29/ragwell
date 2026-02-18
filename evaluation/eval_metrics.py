"""Evaluation metrics for RAG retrieval."""
import math
from typing import List, Dict, Any


def mean_reciprocal_rank(results: List[List[Any]], relevant_ids: List[List[str]]) -> float:
    """Calculate Mean Reciprocal Rank (MRR).
    
    MRR = average of 1/rank of first relevant result
    
    Args:
        results: List of result lists (one per query)
        relevant_ids: List of relevant chunk IDs (one list per query)
        
    Returns:
        MRR score (0-1, higher is better)
    """
    if not results or not relevant_ids:
        return 0.0
    
    rr_sum = 0.0
    
    for query_results, query_relevant in zip(results, relevant_ids):
        # Find first relevant result
        for rank, result in enumerate(query_results, start=1):
            result_id = result.chunk_id if hasattr(result, 'chunk_id') else result
            if result_id in query_relevant:
                rr_sum += 1.0 / rank
                break
    
    return rr_sum / len(results)


def recall_at_k(results: List[List[Any]], relevant_ids: List[List[str]], k: int) -> float:
    """Calculate Recall@K.
    
    Recall@k = fraction of queries where at least one relevant doc is in top-k
    
    Args:
        results: List of result lists
        relevant_ids: List of relevant chunk IDs
        k: Top k results to consider
        
    Returns:
        Recall@k (0-1, higher is better)
    """
    if not results or not relevant_ids:
        return 0.0
    
    hits = 0
    
    for query_results, query_relevant in zip(results, relevant_ids):
        top_k = query_results[:k]
        top_k_ids = [
            r.chunk_id if hasattr(r, 'chunk_id') else r 
            for r in top_k
        ]
        
        # Check if any relevant doc is in top-k
        if any(rel_id in top_k_ids for rel_id in query_relevant):
            hits += 1
    
    return hits / len(results)


def dcg_at_k(relevances: List[float], k: int) -> float:
    """Calculate Discounted Cumulative Gain at K.
    
    DCG@k = sum(rel_i / log2(i + 1)) for i = 1 to k
    
    Args:
        relevances: List of relevance scores (in result order)
        k: Number of results to consider
        
    Returns:
        DCG score
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if i == 1:
            dcg += rel
        else:
            dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    results: List[List[Any]], 
    relevant_ids: List[List[str]], 
    k: int
) -> float:
    """Calculate normalized Discounted Cumulative Gain at K.
    
    nDCG@k = DCG@k / IDCG@k
    where IDCG is the ideal DCG with perfect ranking
    
    Args:
        results: List of result lists
        relevant_ids: List of relevant chunk IDs
        k: Number of results to consider
        
    Returns:
        nDCG score (0-1, higher is better)
    """
    if not results or not relevant_ids:
        return 0.0
    
    ndcg_sum = 0.0
    
    for query_results, query_relevant in zip(results, relevant_ids):
        # Create relevance list for this query
        relevances = []
        for result in query_results:
            result_id = result.chunk_id if hasattr(result, 'chunk_id') else result
            # Binary relevance: 1 if relevant, 0 if not
            relevance = 1.0 if result_id in query_relevant else 0.0
            relevances.append(relevance)
        
        # Calculate DCG
        dcg = dcg_at_k(relevances, k)
        
        # Calculate IDCG (ideal DCG with perfect ranking)
        # All relevant items at top with relevance 1.0
        ideal_relevances = [1.0] * len(query_relevant) + [0.0] * (len(query_results) - len(query_relevant))
        idcg = dcg_at_k(ideal_relevances, k)
        
        # Calculate nDCG
        if idcg > 0:
            ndcg_sum += dcg / idcg
    
    return ndcg_sum / len(results)


def precision_at_k(results: List[List[Any]], relevant_ids: List[List[str]], k: int) -> float:
    """Calculate Precision@K.
    
    Precision@k = average of (# relevant in top-k) / k
    
    Args:
        results: List of result lists
        relevant_ids: List of relevant chunk IDs
        k: Number of results to consider
        
    Returns:
        Precision@k (0-1, higher is better)
    """
    if not results or not relevant_ids:
        return 0.0
    
    precision_sum = 0.0
    
    for query_results, query_relevant in zip(results, relevant_ids):
        top_k = query_results[:k]
        top_k_ids = [
            r.chunk_id if hasattr(r, 'chunk_id') else r 
            for r in top_k
        ]
        
        # Count relevant in top-k
        relevant_count = sum(1 for rel_id in query_relevant if rel_id in top_k_ids)
        precision_sum += relevant_count / k
    
    return precision_sum / len(results)


class MetricsResult:
    """Container for evaluation metrics."""
    
    def __init__(
        self,
        mrr: float,
        recall_1: float,
        recall_3: float,
        recall_5: float,
        recall_10: float,
        ndcg_5: float,
        ndcg_10: float,
        precision_5: float,
        latency_ms: float
    ):
        self.mrr = mrr
        self.recall_1 = recall_1
        self.recall_3 = recall_3
        self.recall_5 = recall_5
        self.recall_10 = recall_10
        self.ndcg_5 = ndcg_5
        self.ndcg_10 = ndcg_10
        self.precision_5 = precision_5
        self.latency_ms = latency_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mrr': round(self.mrr, 4),
            'recall_1': round(self.recall_1, 4),
            'recall_3': round(self.recall_3, 4),
            'recall_5': round(self.recall_5, 4),
            'recall_10': round(self.recall_10, 4),
            'ndcg_5': round(self.ndcg_5, 4),
            'ndcg_10': round(self.ndcg_10, 4),
            'precision_5': round(self.precision_5, 4),
            'latency_ms': round(self.latency_ms, 2)
        }
    
    @classmethod
    def compute(
        cls,
        results: List[List[Any]],
        relevant_ids: List[List[str]],
        latencies_ms: List[float]
    ) -> 'MetricsResult':
        """Compute all metrics for a set of results.
        
        Args:
            results: List of result lists (one per query)
            relevant_ids: List of relevant chunk IDs (one list per query)
            latencies_ms: List of query latencies in milliseconds
            
        Returns:
            MetricsResult with all computed metrics
        """
        return cls(
            mrr=mean_reciprocal_rank(results, relevant_ids),
            recall_1=recall_at_k(results, relevant_ids, 1),
            recall_3=recall_at_k(results, relevant_ids, 3),
            recall_5=recall_at_k(results, relevant_ids, 5),
            recall_10=recall_at_k(results, relevant_ids, 10),
            ndcg_5=ndcg_at_k(results, relevant_ids, 5),
            ndcg_10=ndcg_at_k(results, relevant_ids, 10),
            precision_5=precision_at_k(results, relevant_ids, 5),
            latency_ms=sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        )
