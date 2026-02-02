"""
MLOps Package.

Provides tools for:
- RAG evaluation (RAGAS)
- Experiment tracking (MLflow)
- Data versioning (DVC)
- Pipeline orchestration
"""

from src.mlops.tracking import MLflowTracker, create_tracker
# from src.mlops.evaluation import RAGEvaluator, EvaluationMetric
# from src.mlops.dvc_manager import DVCManager

__all__ = [
    "MLflowTracker",
    "create_tracker",
]
