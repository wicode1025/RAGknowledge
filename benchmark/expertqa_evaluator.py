"""
ExpertQA Benchmark Evaluator
Evaluates RAG system on ExpertQA benchmark dataset.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from rag_system import RAGSystem, RAGResponse
from config import Config

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    hallucination_rate: float
    rejection_rate: float
    avg_similarity: float
    total_questions: int
    answered_questions: int
    rejected_questions: int
    hallucinated_answers: int


@dataclass
class TestCase:
    """ExpertQA test case"""
    question: str
    answer: str
    context: str = ""
    domain: str = ""
    difficulty: str = ""


class ExpertQAEvaluator:
    """
    Evaluator for ExpertQA benchmark
    """

    def __init__(
        self,
        rag_system: RAGSystem = None,
        test_data_path: Path = None
    ):
        """
        Initialize evaluator

        Args:
            rag_system: RAG system to evaluate
            test_data_path: Path to ExpertQA test data
        """
        self.rag_system = rag_system or RAGSystem()
        self.test_data_path = test_data_path or (Config.PROJECT_ROOT / "data" / "expertqa.jsonl")
        self.test_cases: List[TestCase] = []
        self.results: List[Dict] = []

    def load_test_data(self, path: Path = None) -> List[TestCase]:
        """
        Load ExpertQA test data

        Args:
            path: Path to test data file

        Returns:
            List of test cases
        """
        path = path or self.test_data_path

        if not path.exists():
            logger.warning(f"Test data not found at {path}")
            logger.info("Using sample test data for demonstration")
            return self._load_sample_data()

        test_cases = []

        logger.info(f"Loading test data from {path}")

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                test_case = TestCase(
                    question=data.get('question', ''),
                    answer=data.get('answer', ''),
                    context=data.get('context', ''),
                    domain=data.get('domain', ''),
                    difficulty=data.get('difficulty', '')
                )
                test_cases.append(test_case)

        logger.info(f"Loaded {len(test_cases)} test cases")
        self.test_cases = test_cases
        return test_cases

    def _load_sample_data(self) -> List[TestCase]:
        """
        Load sample test data for demonstration

        Returns:
            List of sample test cases
        """
        # Sample ExpertQA-style questions for computer science
        samples = [
            TestCase(
                question="What is the time complexity of quicksort in the average case?",
                answer="O(n log n)",
                domain="Algorithms",
                difficulty="Medium"
            ),
            TestCase(
                question="Explain the difference between process and thread in operating systems.",
                answer="A process is an independent program execution unit with its own memory space, while a thread is a lightweight execution unit within a process that shares memory with other threads.",
                domain="Operating Systems",
                difficulty="Medium"
            ),
            TestCase(
                question="What is the purpose of the attention mechanism in transformers?",
                answer="Attention allows the model to weigh the importance of different parts of the input when processing each element, enabling it to capture long-range dependencies.",
                domain="Deep Learning",
                difficulty="Hard"
            ),
            TestCase(
                question="What is SQL injection and how can it be prevented?",
                answer="SQL injection is an attack that inserts malicious SQL code into queries. Prevention includes using parameterized queries, input validation, and least privilege principles.",
                domain="Cybersecurity",
                difficulty="Medium"
            ),
            TestCase(
                question="Explain the CAP theorem.",
                answer="The CAP theorem states that a distributed system can only guarantee two of three properties: Consistency, Availability, and Partition tolerance. Since network partitions are inevitable, the trade-off is between consistency and availability.",
                context="Database systems",
                difficulty="Hard"
            ),
        ]

        self.test_cases = samples
        return samples

    def evaluate_single(
        self,
        test_case: TestCase,
        check_hallucination: bool = True
    ) -> Dict:
        """
        Evaluate a single test case

        Args:
            test_case: Test case to evaluate
            check_hallucination: Whether to check for hallucinations

        Returns:
            Evaluation result dictionary
        """
        # Query RAG system
        response = self.rag_system.query(test_case.question)

        result = {
            'question': test_case.question,
            'expected_answer': test_case.answer,
            'generated_answer': response.answer,
            'success': response.success,
            'status': response.status.value,
            'max_similarity': response.max_similarity,
            'rejected': not response.success
        }

        # Check hallucination if requested
        if check_hallucination and response.success:
            result['has_hallucination'] = self._check_hallucination(
                test_case.answer,
                response.answer
            )
        else:
            result['has_hallucination'] = False

        return result

    def _check_hallucination(
        self,
        expected: str,
        generated: str,
        threshold: float = 0.3
    ) -> bool:
        """
        Check if generated answer contains hallucination

        Args:
            expected: Expected answer
            generated: Generated answer
            threshold: Similarity threshold for hallucination detection

        Returns:
            True if potential hallucination detected
        """
        # Simple word overlap based detection
        expected_words = set(expected.lower().split())
        generated_words = set(generated.lower().split())

        if not expected_words:
            return False

        # Calculate overlap
        overlap = len(expected_words & generated_words) / len(expected_words)

        # If overlap is too low, might be hallucination
        if overlap < threshold:
            # Additional check: key technical terms
            # If expected answer has technical terms not in generated
            technical_terms = [w for w in expected_words if len(w) > 3]
            missing_terms = [t for t in technical_terms if t not in generated_words]

            # If significant technical terms missing, flag as potential hallucination
            if len(missing_terms) > len(technical_terms) * 0.5:
                return True

        return False

    def evaluate_all(
        self,
        test_cases: List[TestCase] = None,
        check_hallucination: bool = True,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Evaluate all test cases

        Args:
            test_cases: List of test cases (uses loaded if not provided)
            check_hallucination: Whether to check for hallucinations
            show_progress: Whether to show progress

        Returns:
            List of evaluation results
        """
        test_cases = test_cases or self.test_cases

        if not test_cases:
            test_cases = self.load_test_data()

        results = []

        for i, test_case in enumerate(test_cases):
            if show_progress:
                logger.info(f"Evaluating {i+1}/{len(test_cases)}: {test_case.question[:50]}...")

            result = self.evaluate_single(test_case, check_hallucination)
            results.append(result)

        self.results = results
        return results

    def compute_metrics(self) -> EvaluationMetrics:
        """
        Compute evaluation metrics

        Returns:
            EvaluationMetrics object
        """
        if not self.results:
            raise ValueError("No results to evaluate. Run evaluate_all first.")

        total = len(self.results)
        answered = sum(1 for r in self.results if r['success'])
        rejected = total - answered
        hallucinated = sum(1 for r in self.results if r.get('has_hallucination', False))

        # Compute similarity stats
        similarities = [
            r['max_similarity']
            for r in self.results
            if r.get('max_similarity', 0) > 0
        ]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        # Compute rates
        rejection_rate = rejected / total if total > 0 else 0
        hallucination_rate = hallucinated / answered if answered > 0 else 0

        # For ExpertQA, we compute accuracy based on:
        # - Successfully answered AND no hallucination
        accurate = answered - hallucinated
        accuracy = accurate / total if total > 0 else 0

        # Precision: accurate answers / all answered
        precision = accurate / answered if answered > 0 else 0

        # Recall: accurate / total questions
        recall = accurate / total if total > 0 else 0

        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            hallucination_rate=hallucination_rate,
            rejection_rate=rejection_rate,
            avg_similarity=avg_similarity,
            total_questions=total,
            answered_questions=answered,
            rejected_questions=rejected,
            hallucinated_answers=hallucinated
        )

    def compare_with_baseline(
        self,
        baseline_results: List[Dict]
    ) -> Dict:
        """
        Compare RAG results with baseline (direct LLM) results

        Args:
            baseline_results: Results from direct LLM call

        Returns:
            Comparison dictionary
        """
        if not self.results or not baseline_results:
            raise ValueError("Need both RAG and baseline results")

        rag_metrics = self.compute_metrics()

        # Compute baseline metrics
        baseline_hallucinated = sum(
            1 for r in baseline_results if r.get('has_hallucination', False)
        )
        baseline_total = len(baseline_results)
        baseline_hallucination_rate = baseline_hallucinated / baseline_total

        return {
            'rag_accuracy': rag_metrics.accuracy,
            'baseline_accuracy': None,  # Would need ground truth for baseline
            'rag_hallucination_rate': rag_metrics.hallucination_rate,
            'baseline_hallucination_rate': baseline_hallucination_rate,
            'improvement': {
                'hallucination_reduction': baseline_hallucination_rate - rag_metrics.hallucination_rate,
                'rejection_rate': rag_metrics.rejection_rate
            }
        }

    def save_results(self, output_path: Path = None):
        """
        Save evaluation results to file

        Args:
            output_path: Path to save results
        """
        output_path = output_path or (
            Config.PROJECT_ROOT / "evaluation_results.json"
        )

        output_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.compute_metrics().__dict__,
            'results': self.results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_path}")

    def print_summary(self):
        """Print evaluation summary"""
        metrics = self.compute_metrics()

        print("\n" + "="*60)
        print("ExpertQA Evaluation Summary")
        print("="*60)
        print(f"Total Questions:     {metrics.total_questions}")
        print(f"Answered:           {metrics.answered_questions}")
        print(f"Rejected:           {metrics.rejected_questions}")
        print(f"Hallucinated:       {metrics.hallucinated_answers}")
        print("-"*60)
        print(f"Accuracy:           {metrics.accuracy:.4f}")
        print(f"Precision:          {metrics.precision:.4f}")
        print(f"Recall:             {metrics.recall:.4f}")
        print(f"F1 Score:           {metrics.f1:.4f}")
        print(f"Hallucination Rate:  {metrics.hallucination_rate:.4f}")
        print(f"Rejection Rate:     {metrics.rejection_rate:.4f}")
        print(f"Avg Similarity:     {metrics.avg_similarity:.4f}")
        print("="*60)


def run_evaluation(
    test_data_path: Path = None,
    api_key: str = None,
    **rag_kwargs
) -> EvaluationMetrics:
    """
    Run full evaluation

    Args:
        test_data_path: Path to test data
        api_key: Claude API key
        **rag_kwargs: Additional RAG configuration

    Returns:
        EvaluationMetrics
    """
    logging.basicConfig(level=logging.INFO)

    # Create RAG system
    rag = RAGSystem(api_key=api_key, **rag_kwargs)

    # Initialize
    print("Initializing RAG system...")
    rag.initialize()

    # Create evaluator
    evaluator = ExpertQAEvaluator(rag_system=rag)

    # Load test data
    print("Loading test data...")
    evaluator.load_test_data(test_data_path)

    # Run evaluation
    print("Running evaluation...")
    evaluator.evaluate_all()

    # Print summary
    evaluator.print_summary()

    return evaluator.compute_metrics()


if __name__ == "__main__":
    # Run evaluation
    metrics = run_evaluation()
