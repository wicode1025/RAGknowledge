"""
Threshold Filter Module
Evaluates retrieval results and determines if generation should proceed.
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from retrieval.retriever import RetrievalResult
from config import Config

logger = logging.getLogger(__name__)


class RetrievalStatus(Enum):
    """Status of retrieval evaluation"""
    SUCCESS = "success"
    NO_RESULTS = "no_results"
    BELOW_THRESHOLD = "below_threshold"
    EMPTY_QUERY = "empty_query"


@dataclass
class EvaluationResult:
    """Result of evaluation"""
    status: RetrievalStatus
    can_proceed: bool
    message: str
    max_score: float
    result_count: int
    results: List[RetrievalResult]


class ThresholdFilter:
    """
    Filter for evaluating retrieval results against similarity thresholds.
    Implements rejection logic when knowledge base doesn't have relevant info.
    """

    def __init__(
        self,
        threshold: float = None,
        min_results: int = None,
        max_results: int = None
    ):
        """
        Initialize threshold filter

        Args:
            threshold: Minimum similarity score to proceed
            min_results: Minimum number of results required
            max_results: Maximum results to consider
        """
        self.threshold = threshold or Config.SIMILARITY_THRESHOLD
        self.min_results = min_results or 1
        self.max_results = max_results or Config.TOP_K

    def evaluate(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> EvaluationResult:
        """
        Evaluate retrieval results

        Args:
            query: The user query
            results: List of retrieval results

        Returns:
            EvaluationResult with status and decision
        """
        # Handle empty query
        if not query or not query.strip():
            return EvaluationResult(
                status=RetrievalStatus.EMPTY_QUERY,
                can_proceed=False,
                message="Empty query provided",
                max_score=0.0,
                result_count=0,
                results=[]
            )

        # Handle no results
        if not results:
            return EvaluationResult(
                status=RetrievalStatus.NO_RESULTS,
                can_proceed=False,
                message="No relevant results found in knowledge base",
                max_score=0.0,
                result_count=0,
                results=[]
            )

        # Get maximum similarity score
        max_score = max(result.score for result in results)

        # Check if max score meets threshold
        if max_score < self.threshold:
            logger.warning(
                f"Max score {max_score:.4f} below threshold {self.threshold}"
            )
            return EvaluationResult(
                status=RetrievalStatus.BELOW_THRESHOLD,
                can_proceed=False,
                message=f"Knowledge base coverage insufficient (max score: {max_score:.4f}, threshold: {self.threshold})",
                max_score=max_score,
                result_count=len(results),
                results=results
            )

        # Check minimum results requirement
        if len(results) < self.min_results:
            logger.warning(
                f"Only {len(results)} results found, minimum {self.min_results} required"
            )
            return EvaluationResult(
                status=RetrievalStatus.NO_RESULTS,
                can_proceed=False,
                message=f"Insufficient results ({len(results)} < {self.min_results})",
                max_score=max_score,
                result_count=len(results),
                results=[]
            )

        # Success - can proceed with generation
        logger.info(
            f"Evaluation passed: max_score={max_score:.4f}, "
            f"threshold={self.threshold}, results={len(results)}"
        )

        return EvaluationResult(
            status=RetrievalStatus.SUCCESS,
            can_proceed=True,
            message="Evaluation passed",
            max_score=max_score,
            result_count=len(results),
            results=results
        )

    def should_reject(self, results: List[RetrievalResult]) -> Tuple[bool, str]:
        """
        Determine if generation should be rejected

        Args:
            results: Retrieval results

        Returns:
            Tuple of (should_reject, reason)
        """
        if not results:
            return True, "No relevant context found in knowledge base"

        max_score = max(result.score for result in results)

        if max_score < self.threshold:
            return True, (
                f"Context relevance too low (similarity: {max_score:.4f}, "
                f"required: {self.threshold}). The question may be outside "
                f"the knowledge base coverage."
            )

        return False, ""

    def get_rejection_response(self, query: str, max_score: float) -> str:
        """
        Generate a rejection response when knowledge base coverage is insufficient

        Args:
            query: The user's question
            max_score: Maximum similarity score found

        Returns:
            Rejection message
        """
        return (
            "I cannot answer this question based on the available knowledge base. "
            f"The most relevant information found has a similarity score of {max_score:.4f}, "
            "which is below the required threshold. This suggests the question "
            "may be outside the scope of the current knowledge base.\n\n"
            "Please try:\n"
            "1. Rephrasing your question\n"
            "2. Asking about topics covered in the knowledge base\n"
            "3. Adding more relevant documents to the knowledge base"
        )

    def get_confidence_level(self, results: List[RetrievalResult]) -> str:
        """
        Get confidence level based on retrieval scores

        Args:
            results: Retrieval results

        Returns:
            Confidence level: 'high', 'medium', or 'low'
        """
        if not results:
            return 'low'

        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)

        # High confidence: both high average and high max
        if avg_score >= 0.8 and max_score >= 0.85:
            return 'high'
        # Medium confidence: moderate scores
        elif avg_score >= 0.6 and max_score >= 0.7:
            return 'medium'
        else:
            return 'low'


class AdaptiveThresholdFilter(ThresholdFilter):
    """
    Adaptive threshold filter that adjusts based on query characteristics.
    """

    def __init__(
        self,
        base_threshold: float = None,
        min_results: int = None,
        max_results: int = None
    ):
        super().__init__(base_threshold, min_results, max_results)
        self.base_threshold = base_threshold or Config.SIMILARITY_THRESHOLD

    def evaluate(self, query: str, results: List[RetrievalResult]) -> EvaluationResult:
        """
        Evaluate with adaptive threshold based on query type
        """
        # Adjust threshold based on query characteristics
        threshold = self._adjust_threshold(query)

        # Temporarily set adjusted threshold
        original_threshold = self.threshold
        self.threshold = threshold

        # Evaluate
        result = super().evaluate(query, results)

        # Restore original threshold
        self.threshold = original_threshold

        # Update message with actual threshold used
        if not result.can_proceed and result.status == RetrievalStatus.BELOW_THRESHOLD:
            result.message = (
                f"Knowledge base coverage insufficient "
                f"(max score: {result.max_score:.4f}, threshold: {threshold:.4f})"
            )

        return result

    def _adjust_threshold(self, query: str) -> float:
        """
        Adjust threshold based on query characteristics

        Args:
            query: User query

        Returns:
            Adjusted threshold
        """
        # Make threshold more lenient for:
        # - Short queries (may be too specific)
        # - Queries with technical terms (harder to match)

        words = query.split()
        word_count = len(words)

        # Start with base threshold
        threshold = self.base_threshold

        # Adjust for query length
        if word_count <= 3:
            threshold *= 0.9  # More lenient for short queries
        elif word_count >= 10:
            threshold *= 1.05  # Stricter for long queries (more context available)

        # Adjust for technical terms (heuristic: mix of letters and numbers/special chars)
        tech_terms = sum(1 for w in words if any(c.isdigit() for c in w) or '/' in w or '_' in w)
        if tech_terms > 0:
            threshold *= (1 - 0.05 * tech_terms)  # More lenient for technical queries

        return max(0.3, min(0.95, threshold))  # Clamp to reasonable range


class MultiLevelFilter:
    """
    Multi-level filtering that combines multiple criteria.
    """

    def __init__(
        self,
        threshold_filter: ThresholdFilter = None,
        diversity_threshold: float = 0.1
    ):
        self.threshold_filter = threshold_filter or ThresholdFilter()
        self.diversity_threshold = diversity_threshold

    def evaluate(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> EvaluationResult:
        """
        Evaluate with multiple filters
        """
        # First: threshold filter
        eval_result = self.threshold_filter.evaluate(query, results)

        if not eval_result.can_proceed:
            return eval_result

        # Second: diversity check
        if len(results) > 1:
            diversity_score = self._calculate_diversity(results)

            if diversity_score < self.diversity_threshold:
                # Results are too similar - may indicate redundant context
                logger.warning(
                    f"Low diversity score: {diversity_score:.4f}. "
                    "Results may be too similar."
                )

        return eval_result

    def _calculate_diversity(self, results: List[RetrievalResult]) -> float:
        """
        Calculate diversity score based on text overlap

        Args:
            results: List of retrieval results

        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(results) <= 1:
            return 1.0

        scores = []
        for i, r1 in enumerate(results):
            min_overlap = float('inf')
            for r2 in results[i+1:]:
                overlap = self._text_overlap(r1.text, r2.text)
                min_overlap = min(min_overlap, overlap)
            scores.append(1 - min_overlap)

        return sum(scores) / len(scores) if scores else 0.0

    def _text_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


if __name__ == "__main__":
    # Test threshold filter
    logging.basicConfig(level=logging.INFO)

    filter = ThresholdFilter(threshold=0.7, min_results=1)

    # Test cases
    test_results = [
        RetrievalResult(text="Test 1", score=0.85, metadata={}, rank=1),
        RetrievalResult(text="Test 2", score=0.75, metadata={}, rank=2),
    ]

    low_results = [
        RetrievalResult(text="Test 1", score=0.45, metadata={}, rank=1),
    ]

    # Test evaluation
    result = filter.evaluate("What is deep learning?", test_results)
    print(f"High results: {result.status}, proceed: {result.can_proceed}")

    result = filter.evaluate("What is deep learning?", low_results)
    print(f"Low results: {result.status}, proceed: {result.can_proceed}")
    print(f"Message: {result.message}")
