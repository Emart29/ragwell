"""Quality validator for text chunks."""
from dataclasses import dataclass
from typing import Optional
import re
from app.chunking.recursive import Chunk


@dataclass
class QualityResult:
    """Quality check result for a chunk."""
    score: float  # 0.0 to 1.0
    passed: bool  # True if score >= 0.5
    issues: list[str]  # List of quality issues found


class ChunkValidator:
    """Validator for chunk quality.
    
    Scores chunks from 0.0 to 1.0 based on various quality metrics.
    Chunks with score >= 0.5 are considered passed.
    """
    
    def __init__(self, pass_threshold: float = 0.5):
        """Initialize the validator.
        
        Args:
            pass_threshold: Minimum score to pass validation
        """
        self.pass_threshold = pass_threshold
    
    def validate(self, chunk: Chunk) -> QualityResult:
        """Validate a chunk and return quality score.
        
        Args:
            chunk: Chunk to validate
            
        Returns:
            QualityResult with score, pass/fail, and issues
        """
        issues = []
        score = 1.0
        
        text = chunk.text
        token_count = chunk.token_count
        
        # Check 1: Minimum token count (< 20 tokens = reject)
        if token_count < 20:
            return QualityResult(
                score=0.0,
                passed=False,
                issues=["Chunk too small: < 20 tokens"]
            )
        
        # Check 2: Whitespace ratio
        whitespace_score, whitespace_issue = self._check_whitespace(text)
        score *= whitespace_score
        if whitespace_issue:
            issues.append(whitespace_issue)
        
        # Check 3: Encoding artifacts
        encoding_score, encoding_issue = self._check_encoding_artifacts(text)
        score *= encoding_score
        if encoding_issue:
            issues.append(encoding_issue)
        
        # Check 4: Repetitive content
        repetition_score, repetition_issue = self._check_repetition(text)
        score *= repetition_score
        if repetition_issue:
            issues.append(repetition_issue)
        
        # Check 5: Coherent language (not mostly numbers/symbols)
        coherence_score, coherence_issue = self._check_coherence(text)
        score *= coherence_score
        if coherence_issue:
            issues.append(coherence_issue)
        
        return QualityResult(
            score=round(score, 2),
            passed=score >= self.pass_threshold,
            issues=issues
        )
    
    def validate_batch(self, chunks: list[Chunk]) -> list[QualityResult]:
        """Validate multiple chunks.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            List of QualityResults
        """
        return [self.validate(chunk) for chunk in chunks]
    
    def _check_whitespace(self, text: str) -> tuple[float, Optional[str]]:
        """Check whitespace ratio.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (score, issue or None)
        """
        if not text:
            return 0.0, "Empty text"
        
        whitespace_ratio = text.count(' ') / len(text)
        
        if whitespace_ratio > 0.5:
            return 0.5, f"High whitespace ratio: {whitespace_ratio:.1%}"
        elif whitespace_ratio > 0.4:
            return 0.7, f"Elevated whitespace ratio: {whitespace_ratio:.1%}"
        
        return 1.0, None
    
    def _check_encoding_artifacts(self, text: str) -> tuple[float, Optional[str]]:
        """Check for encoding artifacts.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (score, issue or None)
        """
        issues = []
        
        # Check for null bytes
        null_count = text.count('\x00')
        if null_count > 0:
            issues.append(f"{null_count} null bytes")
        
        # Check for replacement characters
        replacement_count = text.count('\ufffd')
        if replacement_count > 0:
            issues.append(f"{replacement_count} replacement characters")
        
        # Check for control characters (excluding standard whitespace)
        control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
        if control_chars > 5:
            issues.append(f"{control_chars} control characters")
        
        if issues:
            # Penalty based on number of issues
            penalty = min(0.5, len(issues) * 0.15)
            return 1.0 - penalty, f"Encoding artifacts: {', '.join(issues)}"
        
        return 1.0, None
    
    def _check_repetition(self, text: str) -> tuple[float, Optional[str]]:
        """Check for repetitive content using 5-grams.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (score, issue or None)
        """
        words = re.findall(r'\b\w+\b', text.lower())
        
        if len(words) < 5:
            return 1.0, None
        
        # Create 5-grams
        ngrams = []
        for i in range(len(words) - 4):
            ngram = ' '.join(words[i:i+5])
            ngrams.append(ngram)
        
        # Count occurrences
        from collections import Counter
        ngram_counts = Counter(ngrams)
        
        # Find repeats
        max_repeats = max(ngram_counts.values())
        
        if max_repeats > 5:
            return 0.3, f"Excessive repetition: 5-gram repeats {max_repeats} times"
        elif max_repeats > 3:
            return 0.7, f"Notable repetition: 5-gram repeats {max_repeats} times"
        
        return 1.0, None
    
    def _check_coherence(self, text: str) -> tuple[float, Optional[str]]:
        """Check if text contains coherent language (not mostly numbers/symbols).
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (score, issue or None)
        """
        if not text:
            return 0.0, "Empty text"
        
        total_chars = len(text)
        if total_chars == 0:
            return 0.0, "Empty text"
        
        # Count letter characters
        letter_count = sum(1 for c in text if c.isalpha())
        letter_ratio = letter_count / total_chars
        
        # Count digit characters
        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digit_count / total_chars
        
        # Count symbol characters (not alphanumeric or whitespace)
        symbol_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        symbol_ratio = symbol_count / total_chars
        
        issues = []
        
        # Check letter ratio
        if letter_ratio < 0.3:
            issues.append(f"Low letter ratio: {letter_ratio:.1%}")
        
        # Check digit ratio
        if digit_ratio > 0.5:
            issues.append(f"High digit ratio: {digit_ratio:.1%}")
        
        # Check symbol ratio
        if symbol_ratio > 0.3:
            issues.append(f"High symbol ratio: {symbol_ratio:.1%}")
        
        if issues:
            penalty = min(0.6, len(issues) * 0.25)
            return 1.0 - penalty, f"Low coherence: {', '.join(issues)}"
        
        return 1.0, None
    
    def get_validation_summary(
        self,
        chunks: list[Chunk],
        results: list[QualityResult]
    ) -> dict:
        """Get summary statistics for batch validation.
        
        Args:
            chunks: List of chunks
            results: List of validation results
            
        Returns:
            Dictionary with summary statistics
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        
        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Collect all issues
        all_issues = []
        for r in results:
            all_issues.extend(r.issues)
        
        from collections import Counter
        issue_counts = Counter(all_issues)
        top_issues = issue_counts.most_common(5)
        
        return {
            'total_chunks': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': round(passed / total * 100, 1) if total > 0 else 0,
            'average_score': round(avg_score, 2),
            'min_score': round(min(scores), 2) if scores else 0,
            'max_score': round(max(scores), 2) if scores else 0,
            'top_issues': top_issues
        }
