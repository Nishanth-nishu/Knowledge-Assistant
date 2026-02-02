"""
RAGAS Evaluation Module.

Provides automated evaluation of RAG pipelines using RAGAS metrics:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall
"""

import logging
from typing import Any

import pandas as pd
from datasets import Dataset

try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    
from src.mlops.tracking import MLflowTracker

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluator for RAG pipelines using RAGAS.
    """
    
    def __init__(self, tracker: MLflowTracker | None = None):
        """
        Initialize evaluator.
        
        Args:
            tracker: Optional MLflow tracker to log results
        """
        self.tracker = tracker
        
        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS not available. Evaluation will be disabled.")
    
    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[list[str]] | None = None,
        run_name: str = "rag_eval",
    ) -> dict[str, float]:
        """
        Run RAGAS evaluation.
        
        Args:
            questions: List of user queries
            answers: List of generated answers
            contexts: List of retrieved contexts (list of strings) per query
            ground_truths: Optional list of ground truth answers
            run_name: Name for the MLflow run
        
        Returns:
            Dictionary of metric scores
        """
        if not RAGAS_AVAILABLE:
            logger.error("RAGAS not installed. Cannot evaluate.")
            return {}
        
        # Prepare dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
        ]
        
        if ground_truths:
            data["ground_truth"] = ground_truths
            metrics.append(context_recall)
        
        dataset = Dataset.from_dict(data)
        
        logger.info(f"Starting RAGAS evaluation on {len(questions)} samples...")
        
        # Run evaluation
        try:
            results = evaluate(
                dataset=dataset,
                metrics=metrics,
            )
            
            # Convert to dict
            scores = results
            
            logger.info(f"Evaluation complete: {scores}")
            
            # Log to MLflow if tracker provided
            if self.tracker:
                with self.tracker.start_run(run_name=run_name):
                    self.tracker.log_metrics(scores)
                    self.tracker.log_params({"samples": len(questions)})
                    
                    # Log dataframe as artifact
                    df = results.to_pandas()
                    df.to_csv("eval_results.csv", index=False)
                    self.tracker.log_artifact("eval_results.csv")
            
            return scores
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {}


__all__ = ["RAGEvaluator"]
