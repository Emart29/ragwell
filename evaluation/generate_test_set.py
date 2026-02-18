"""Generate test dataset for RAG evaluation."""
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import logging

from app.storage.database import DatabaseManager
from app.storage.models import Chunk
from app.config import settings
from app.llm import provider


logger = logging.getLogger(__name__)

# Default test queries if Gemini is not available
DEFAULT_TEST_QUERIES = [
    {
        "query": "What is machine learning?",
        "relevant_keywords": ["machine learning", "ML", "algorithms"]
    },
    {
        "query": "How does neural networks work?",
        "relevant_keywords": ["neural network", "deep learning", "layers"]
    },
    {
        "query": "What is natural language processing?",
        "relevant_keywords": ["NLP", "natural language", "text processing"]
    },
    {
        "query": "Explain document retrieval",
        "relevant_keywords": ["retrieval", "search", "documents"]
    },
    {
        "query": "What are embeddings?",
        "relevant_keywords": ["embedding", "vector", "semantic"]
    },
    {
        "query": "How to chunk documents?",
        "relevant_keywords": ["chunk", "split", "text segmentation"]
    },
    {
        "query": "What is vector similarity?",
        "relevant_keywords": ["similarity", "cosine", "distance"]
    },
    {
        "query": "How does full-text search work?",
        "relevant_keywords": ["search", "FTS5", "keyword"]
    },
    {
        "query": "What is RAG?",
        "relevant_keywords": ["RAG", "retrieval", "generation"]
    },
    {
        "query": "How to evaluate retrieval systems?",
        "relevant_keywords": ["evaluation", "metrics", "MRR", "recall"]
    }
]


class TestDatasetGenerator:
    """Generate test queries from stored chunks."""
    
    def __init__(self, output_path: str = "evaluation/test_queries.json"):
        """Initialize generator.
        
        Args:
            output_path: Path to save test queries
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Check LLM availability once at the start
        self.llm_available = self._check_llm_available()
    
    def _check_llm_available(self) -> bool:
        """Check if any LLM provider is available and has a valid key.
        
        This validates the provider once to avoid repeated failures.
        """
        # We try a simple generation to verify the provider works
        try:
            test_response = provider.generate("Respond with 'ok'", max_tokens=10)
            if test_response:
                logger.info("LLM provider validated successfully.")
                return True
        except Exception as e:
            logger.info(f"LLM provider validation failed ({str(e)}). Using heuristic fallback.")
        
        logger.info("No valid LLM provider found or validation failed. Using heuristic fallback.")
        return False
    
    def generate(self, num_queries: int = 15) -> List[Dict[str, Any]]:
        """Generate test queries.
        
        Args:
            num_queries: Number of queries to generate
            
        Returns:
            List of test queries with relevant chunk IDs
        """
        # Get chunks from database and generate queries within the same session
        with DatabaseManager() as db:
            chunks = db.query(Chunk).all()
            
            if not chunks:
                logger.warning("No chunks found in database. Using default test queries.")
                return self._create_default_queries()
            
            # Sample chunks
            sample_size = min(num_queries, len(chunks))
            sampled_chunks = random.sample(chunks, sample_size)
            
            if self.llm_available:
                return self._generate_with_llm(sampled_chunks)
            else:
                return self._generate_without_llm(sampled_chunks)
    
    def _generate_with_llm(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """Generate queries using LLM.
        
        Args:
            chunks: Chunks to generate queries for
            
        Returns:
            List of test queries
        """
        queries = []
        
        for chunk in chunks:
            try:
                # Generate question
                prompt = f"""Given the following text passage, generate a natural question that this passage would answer:

Text: {chunk.text[:500]}

Generate a single, specific question that someone might ask to find this information:"""
                
                query = provider.generate(prompt)
                if not query:
                    raise ValueError("LLM returned empty response")
                
                query = query.strip()
                
                # Extract keywords from the chunk
                keywords = self._extract_keywords(chunk.text)
                
                queries.append({
                    "query": query,
                    "relevant_chunk_ids": [chunk.id],
                    "relevant_keywords": keywords
                })
                
            except Exception as e:
                logger.warning(f"LLM generation failed for chunk {chunk.id}: {e}. Using heuristic fallback.")
                # Use heuristic fallback
                keywords = self._extract_keywords(chunk.text)
                query = self._create_natural_question(chunk.text, keywords)
                queries.append({
                    "query": query,
                    "relevant_chunk_ids": [chunk.id],
                    "relevant_keywords": keywords
                })
        
        return queries
    
    def _generate_without_llm(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """Generate queries without LLM (create natural questions).
        
        Args:
            chunks: Chunks to generate queries for
            
        Returns:
            List of test queries
        """
        import re
        queries = []
        
        for chunk in chunks:
            # Extract key phrases
            keywords = self._extract_keywords(chunk.text)
            
            # Create natural questions based on content type and keywords
            query = self._create_natural_question(chunk.text, keywords)
            
            queries.append({
                "query": query,
                "relevant_chunk_ids": [chunk.id],
                "relevant_keywords": keywords
            })
        
        return queries
    
    def _create_natural_question(self, text: str, keywords: List[str]) -> str:
        """Create a natural question from chunk text and keywords."""
        import re
        
        # Detect content type patterns
        text_lower = text.lower()
        
        # CSV/Data patterns - look for structured data indicators
        if any(word in text_lower for word in ['sales', 'revenue', 'target', 'region', 'id', 'txn-', 'performance']):
            # Extract specific values from CSV-like content
            numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)
            regions = re.findall(r'\b(?:North America|Europe|Asia Pacific|Latin America|North|South|East|West)\b', text, re.IGNORECASE)
            
            if numbers and regions:
                return f"What were the sales figures for {regions[0]} region?"
            elif 'target' in text_lower and numbers:
                return f"Which regions exceeded their sales targets of {numbers[0] if numbers else '1500'}?"
            elif 'performance' in text_lower:
                return "How did different regions perform against their sales targets?"
            elif keywords:
                return f"What are the {keywords[0]} metrics by region?"
            return "What are the sales performance results by geographic region?"
        
        # Meeting notes patterns - look for meeting-specific content
        if any(word in text_lower for word in ['action', 'item', 'meeting', 'discuss', 'decision', 'sync', 'minutes']):
            if 'action' in text_lower and 'item' in text_lower:
                return "What action items were identified during the meeting?"
            elif 'decision' in text_lower:
                return "What key decisions were made in the meeting?"
            elif 'implement' in text_lower:
                return "What implementation tasks were discussed?"
            elif 'optimize' in text_lower:
                return "What optimization strategies were proposed?"
            else:
                return "What were the main discussion points covered in the meeting?"
        
        # Technical/Architecture patterns - look for system-related content
        if any(word in text_lower for word in ['system', 'architecture', 'implementation', 'performance', 'ragwell', 'retrieval']):
            if 'performance' in text_lower and 'benchmark' in text_lower:
                return "What performance benchmarks were achieved by the system?"
            elif 'architecture' in text_lower and 'layer' in text_lower:
                return "What are the key architectural layers of the system?"
            elif 'retrieval' in text_lower and 'hybrid' in text_lower:
                return "How does the hybrid retrieval mechanism work?"
            elif 'scalability' in text_lower:
                return "What scalability features does the system provide?"
            elif keywords and len(keywords) >= 2:
                return f"How does {keywords[0]} integrate with {keywords[1]} in the system?"
            return "What are the main technical capabilities of the system?"
        
        # Financial/Business patterns - look for business content
        if any(word in text_lower for word in ['report', 'summary', 'analysis', 'strategy', 'enterprise', 'customer']):
            if 'strategic' in text_lower and 'pivot' in text_lower:
                return "What strategic changes are being implemented?"
            elif 'enterprise' in text_lower and 'customer' in text_lower:
                return "What enterprise customer requirements are addressed?"
            elif 'analysis' in text_lower:
                return "What key insights were revealed in the analysis?"
            elif keywords:
                return f"What does the business report indicate about {keywords[0]}?"
            return "What are the main business findings discussed in the report?"
        
        # Security/Compliance patterns - look for security content
        if any(word in text_lower for word in ['security', 'compliance', 'access', 'privacy', 'rbac', 'audit']):
            if 'rbac' in text_lower or 'access control' in text_lower:
                return "What access control mechanisms are implemented?"
            elif 'compliance' in text_lower and 'audit' in text_lower:
                return "What compliance and audit features are available?"
            elif 'privacy' in text_lower and 'data' in text_lower:
                return "How is data privacy protected in the system?"
            return "What security measures are implemented?"
        
        # Extract specific metrics and ask about them
        percentages = re.findall(r'\b\d+(?:\.\d+)?%\b', text)
        if percentages and keywords:
            return f"What is the significance of the {percentages[0]} improvement in {keywords[0]}?"
        
        # Look for time-based information
        time_patterns = re.findall(r'\b(?:Q[1-4]|quarter|annual|monthly|daily)\b', text, re.IGNORECASE)
        if time_patterns and keywords:
            return f"What are the {time_patterns[0].lower()} trends for {keywords[0]}?"
        
        # Look for comparison patterns
        if any(word in text_lower for word in ['compare', 'versus', 'better', 'improve', 'exceed', 'outperform']):
            if keywords and len(keywords) >= 2:
                return f"How does {keywords[0]} compare to {keywords[1]}?"
            elif 'outperform' in text_lower or 'exceed' in text_lower:
                return "What performance improvements were achieved?"
            return "What comparisons or improvements are highlighted?"
        
        # Look for process or methodology descriptions
        if any(word in text_lower for word in ['process', 'method', 'approach', 'strategy', 'technique']):
            if keywords:
                return f"What is the methodology used for {keywords[0]}?"
            return "What processes or methodologies are described?"
        
        # Extract entities and create relationship questions
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        tech_entities = [e for e in entities if any(tech in e.lower() for tech in ['rag', 'sql', 'api', 'db', 'fts', 'bm25'])]
        
        if tech_entities and len(tech_entities) > 1:
            return f"How do {tech_entities[0]} and {tech_entities[1]} work together?"
        elif entities and len(entities) > 1:
            return f"What is the relationship between {entities[0]} and {entities[1]}?"
        
        # Generic question based on keywords with better context
        if keywords:
            if len(keywords) >= 3:
                return f"How do {keywords[0]}, {keywords[1]}, and {keywords[2]} relate to each other?"
            elif len(keywords) >= 2:
                return f"What is the connection between {keywords[0]} and {keywords[1]}?"
            else:
                return f"What key information is provided about {keywords[0]}?"
        
        # Fallback: extract meaningful phrases and create questions
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 50]
        if sentences:
            sentence = sentences[0]
            # Extract the main subject (first few meaningful words)
            words = [w for w in sentence.split()[:10] if len(w) > 3][:5]
            if words:
                return f"What details are provided about {' '.join(words).lower()}?"
        
        return "What is the main information presented in this content?"
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction: find most frequent non-stop words
        import re
        from collections import Counter
        
        # Common stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'and', 'but', 'or', 'yet',
            'so', 'if', 'because', 'although', 'though', 'while', 'where',
            'when', 'that', 'which', 'who', 'whom', 'whose', 'what', 'this',
            'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
            'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
            'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
            'about', 'against', 'between', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
            'then', 'once'
        }
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stop words
        words = [w for w in words if w not in stop_words]
        
        # Count and get most common
        word_counts = Counter(words)
        most_common = word_counts.most_common(max_keywords)
        
        return [word for word, count in most_common]
    
    def _create_default_queries(self) -> List[Dict[str, Any]]:
        """Create default test queries when no chunks available.
        
        Returns:
            List of default queries
        """
        return DEFAULT_TEST_QUERIES
    
    def save(self, queries: List[Dict[str, Any]]) -> str:
        """Save queries to file.
        
        Args:
            queries: List of queries to save
            
        Returns:
            Path to saved file
        """
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(queries)} test queries to {self.output_path}")
        return str(self.output_path)
    
    def load(self) -> List[Dict[str, Any]]:
        """Load queries from file.
        
        Returns:
            List of queries
        """
        if not self.output_path.exists():
            # Generate if not exists
            queries = self.generate()
            self.save(queries)
            return queries
        
        with open(self.output_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def generate_test_set(num_queries: int = 15) -> str:
    """Generate test dataset and save to file.
    
    Args:
        num_queries: Number of queries to generate
        
    Returns:
        Path to saved file
    """
    generator = TestDatasetGenerator()
    queries = generator.generate(num_queries)
    return generator.save(queries)


if __name__ == "__main__":
    path = generate_test_set()
    print(f"Generated test set: {path}")
