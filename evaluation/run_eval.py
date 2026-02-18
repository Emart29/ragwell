"""Evaluation runner for RAG retrieval."""
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

from evaluation.eval_metrics import MetricsResult
from evaluation.generate_test_set import TestDatasetGenerator
from app.retrieval.retriever import Retriever


logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Run evaluation on retrieval strategies."""
    
    STRATEGIES = ['vector', 'keyword', 'hybrid']
    
    def __init__(self, results_path: str = "evaluation/results.json"):
        """Initialize evaluation runner.
        
        Args:
            results_path: Path to save evaluation results
        """
        self.results_path = Path(results_path)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.retriever = Retriever()
    
    def run(
        self,
        test_queries_path: str = "evaluation/test_queries.json",
        top_k: int = 20
    ) -> Dict[str, Any]:
        """Run evaluation on all strategies.
        
        Args:
            test_queries_path: Path to test queries file
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with evaluation results for all strategies
        """
        # Load test queries
        generator = TestDatasetGenerator(test_queries_path)
        test_queries = generator.load()
        
        logger.info(f"Running evaluation on {len(test_queries)} queries")
        
        # Run evaluation for each strategy
        results = {}
        
        for strategy in self.STRATEGIES:
            logger.info(f"Evaluating strategy: {strategy}")
            results[strategy] = self._evaluate_strategy(strategy, test_queries, top_k)
        
        # Add strategy with re-ranking
        logger.info("Evaluating strategy: hybrid+rerank")
        results['hybrid+rerank'] = self._evaluate_strategy(
            'hybrid', test_queries, top_k, rerank=True
        )
        
        # Save results
        self._save_results(results)
        
        # Print comparison table
        self._print_comparison(results)
        
        return results
    
    def _evaluate_strategy(
        self,
        strategy: str,
        test_queries: List[Dict[str, Any]],
        top_k: int,
        rerank: bool = False
    ) -> Dict[str, Any]:
        """Evaluate a single strategy.
        
        Args:
            strategy: Strategy name
            test_queries: Test queries
            top_k: Number of results
            rerank: Whether to re-rank
            
        Returns:
            Evaluation results
        """
        all_results = []
        all_relevant_ids = []
        latencies = []
        per_query_results = []
        
        for query_data in test_queries:
            query = query_data['query']
            relevant_ids = query_data.get('relevant_chunk_ids', [])
            
            # Run search
            start_time = time.time()
            
            if rerank:
                retrieval_result = self.retriever.retrieve(
                    query=query,
                    strategy=strategy,
                    top_k=top_k,
                    rerank=True,
                    rerank_top_n=top_k
                )
            else:
                retrieval_result = self.retriever.retrieve(
                    query=query,
                    strategy=strategy,
                    top_k=top_k,
                    rerank=False
                )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Store results
            all_results.append(retrieval_result.results)
            all_relevant_ids.append(relevant_ids)
            latencies.append(latency_ms)
            
            # Per-query results
            per_query_results.append({
                'query': query,
                'results_count': len(retrieval_result.results),
                'latency_ms': round(latency_ms, 2)
            })
        
        # Compute metrics
        metrics = MetricsResult.compute(all_results, all_relevant_ids, latencies)
        
        return {
            'strategy': f"{strategy}{'+rerank' if rerank else ''}",
            'metrics': metrics.to_dict(),
            'per_query': per_query_results
        }
    
    def _save_results(self, results: Dict[str, Any]) -> str:
        """Save evaluation results to file.
        
        Args:
            results: Evaluation results
            
        Returns:
            Path to saved file
        """
        with open(self.results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {self.results_path}")
        return str(self.results_path)
    
    def _print_comparison(self, results: Dict[str, Any]):
        """Print comparison table.
        
        Args:
            results: Evaluation results
        """
        print("\n" + "=" * 100)
        print("RAGWELL EVALUATION RESULTS")
        print("=" * 100)
        print(f"{'Strategy':<20} {'MRR':>8} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'nDCG@5':>8} {'nDCG@10':>8} {'Latency':>10}")
        print("-" * 100)
        
        for strategy_name, data in results.items():
            metrics = data['metrics']
            print(
                f"{strategy_name:<20} "
                f"{metrics['mrr']:>8.4f} "
                f"{metrics['recall_1']:>8.4f} "
                f"{metrics['recall_5']:>8.4f} "
                f"{metrics['recall_10']:>8.4f} "
                f"{metrics['ndcg_5']:>8.4f} "
                f"{metrics['ndcg_10']:>8.4f} "
                f"{metrics['latency_ms']:>9.1f}ms"
            )
        
        print("=" * 100 + "\n")
    
    def load_results(self) -> Dict[str, Any]:
        """Load evaluation results from file.
        
        Returns:
            Evaluation results
        """
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
        
        with open(self.results_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def run_evaluation(
    test_queries_path: str = "evaluation/test_queries.json",
    results_path: str = "evaluation/results.json"
) -> Dict[str, Any]:
    """Run evaluation and return results.
    
    Args:
        test_queries_path: Path to test queries
        results_path: Path to save results
        
    Returns:
        Evaluation results
    """
    runner = EvaluationRunner(results_path)
    return runner.run(test_queries_path)


if __name__ == "__main__":
    results = run_evaluation()
    print("\nEvaluation complete!")
