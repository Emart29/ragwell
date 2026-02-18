"""Chunking strategy comparator."""
from dataclasses import dataclass
from typing import Optional
from app.parsers.base import ParsedDocument
from app.chunking.recursive import RecursiveChunker, Chunk
from app.chunking.semantic import SemanticChunker


@dataclass
class ChunkingComparison:
    """Comparison results between chunking strategies."""
    # Recursive strategy stats
    recursive_chunk_count: int
    recursive_avg_size: float
    recursive_min_size: int
    recursive_max_size: int
    recursive_distribution: dict
    recursive_sample_chunks: list[str]
    
    # Semantic strategy stats
    semantic_chunk_count: int
    semantic_avg_size: float
    semantic_min_size: int
    semantic_max_size: int
    semantic_distribution: dict
    semantic_sample_chunks: list[str]
    
    # Comparison summary
    size_difference_percent: float
    recommendation: str


class ChunkingComparator:
    """Compare different chunking strategies on a document."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """Initialize comparator with chunking parameters.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def compare(self, parsed_doc: ParsedDocument) -> ChunkingComparison:
        """Run both chunking strategies and compare results.
        
        Args:
            parsed_doc: Parsed document to chunk
            
        Returns:
            ChunkingComparison with detailed stats
        """
        # Run recursive chunking
        recursive_chunker = RecursiveChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        recursive_chunks = recursive_chunker.chunk_document(parsed_doc)
        
        # Run semantic chunking
        semantic_chunker = SemanticChunker(max_tokens=self.chunk_size)
        semantic_chunks = semantic_chunker.chunk_document(parsed_doc)
        
        # Calculate statistics
        recursive_stats = self._calculate_stats(recursive_chunks)
        semantic_stats = self._calculate_stats(semantic_chunks)
        
        # Calculate size difference
        size_diff = (
            (semantic_stats['count'] - recursive_stats['count']) 
            / recursive_stats['count'] * 100
            if recursive_stats['count'] > 0 else 0
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            recursive_stats, semantic_stats
        )
        
        return ChunkingComparison(
            recursive_chunk_count=recursive_stats['count'],
            recursive_avg_size=recursive_stats['avg_size'],
            recursive_min_size=recursive_stats['min_size'],
            recursive_max_size=recursive_stats['max_size'],
            recursive_distribution=recursive_stats['distribution'],
            recursive_sample_chunks=recursive_stats['samples'],
            semantic_chunk_count=semantic_stats['count'],
            semantic_avg_size=semantic_stats['avg_size'],
            semantic_min_size=semantic_stats['min_size'],
            semantic_max_size=semantic_stats['max_size'],
            semantic_distribution=semantic_stats['distribution'],
            semantic_sample_chunks=semantic_stats['samples'],
            size_difference_percent=round(size_diff, 2),
            recommendation=recommendation
        )
    
    def _calculate_stats(self, chunks: list[Chunk]) -> dict:
        """Calculate statistics for a list of chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                'count': 0,
                'avg_size': 0.0,
                'min_size': 0,
                'max_size': 0,
                'distribution': {},
                'samples': []
            }
        
        sizes = [chunk.token_count for chunk in chunks]
        
        # Calculate distribution buckets
        distribution = self._calculate_distribution(sizes)
        
        # Get sample chunks (first 3)
        samples = [chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text 
                   for chunk in chunks[:3]]
        
        return {
            'count': len(chunks),
            'avg_size': round(sum(sizes) / len(sizes), 2),
            'min_size': min(sizes),
            'max_size': max(sizes),
            'distribution': distribution,
            'samples': samples
        }
    
    def _calculate_distribution(self, sizes: list[int]) -> dict:
        """Calculate chunk size distribution in buckets.
        
        Args:
            sizes: List of chunk sizes in tokens
            
        Returns:
            Dictionary with bucket counts
        """
        buckets = {
            '0-100': 0,
            '101-200': 0,
            '201-300': 0,
            '301-400': 0,
            '401-500': 0,
            '501-600': 0,
            '601+': 0
        }
        
        for size in sizes:
            if size <= 100:
                buckets['0-100'] += 1
            elif size <= 200:
                buckets['101-200'] += 1
            elif size <= 300:
                buckets['201-300'] += 1
            elif size <= 400:
                buckets['301-400'] += 1
            elif size <= 500:
                buckets['401-500'] += 1
            elif size <= 600:
                buckets['501-600'] += 1
            else:
                buckets['601+'] += 1
        
        # Convert to percentages
        total = len(sizes)
        return {k: round(v / total * 100, 1) for k, v in buckets.items()}
    
    def _generate_recommendation(
        self,
        recursive_stats: dict,
        semantic_stats: dict
    ) -> str:
        """Generate recommendation based on comparison.
        
        Args:
            recursive_stats: Recursive chunking statistics
            semantic_stats: Semantic chunking statistics
            
        Returns:
            Recommendation string
        """
        rec_score = 0
        reasons = []
        
        # Check chunk count difference
        count_diff = abs(semantic_stats['count'] - recursive_stats['count'])
        if count_diff / recursive_stats['count'] > 0.3:
            rec_score += 1
            reasons.append("significant chunk count difference")
        
        # Check size consistency
        recursive_range = recursive_stats['max_size'] - recursive_stats['min_size']
        semantic_range = semantic_stats['max_size'] - semantic_stats['min_size']
        
        if semantic_range < recursive_range:
            rec_score += 1
            reasons.append("semantic has more consistent sizes")
        
        # Check for very small chunks
        if semantic_stats['min_size'] >= 50:
            rec_score += 1
            reasons.append("semantic avoids very small chunks")
        
        # Generate recommendation
        if rec_score >= 2:
            return f"RECOMMEND SEMANTIC: {', '.join(reasons)}"
        elif semantic_stats['count'] < recursive_stats['count'] * 0.8:
            return f"RECOMMEND RECURSIVE: semantic produced too few chunks"
        else:
            return "SIMILAR PERFORMANCE: either strategy acceptable"
    
    def print_comparison(self, comparison: ChunkingComparison):
        """Print formatted comparison report.
        
        Args:
            comparison: ChunkingComparison to print
        """
        print("\n" + "=" * 60)
        print("CHUNKING STRATEGY COMPARISON")
        print("=" * 60)
        
        print("\n--- Recursive Character Chunking ---")
        print(f"  Total chunks: {comparison.recursive_chunk_count}")
        print(f"  Average size: {comparison.recursive_avg_size} tokens")
        print(f"  Size range: {comparison.recursive_min_size} - {comparison.recursive_max_size} tokens")
        print(f"  Distribution:")
        for bucket, pct in comparison.recursive_distribution.items():
            print(f"    {bucket}: {pct}%")
        print(f"  Sample chunks:")
        for i, sample in enumerate(comparison.recursive_sample_chunks, 1):
            print(f"    {i}. {sample[:100]}...")
        
        print("\n--- Semantic Boundary Chunking ---")
        print(f"  Total chunks: {comparison.semantic_chunk_count}")
        print(f"  Average size: {comparison.semantic_avg_size} tokens")
        print(f"  Size range: {comparison.semantic_min_size} - {comparison.semantic_max_size} tokens")
        print(f"  Distribution:")
        for bucket, pct in comparison.semantic_distribution.items():
            print(f"    {bucket}: {pct}%")
        print(f"  Sample chunks:")
        for i, sample in enumerate(comparison.semantic_sample_chunks, 1):
            print(f"    {i}. {sample[:100]}...")
        
        print("\n--- Summary ---")
        print(f"  Size difference: {comparison.size_difference_percent:+.1f}%")
        print(f"  {comparison.recommendation}")
        print("=" * 60 + "\n")
