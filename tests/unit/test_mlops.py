"""
Tests for MLOps components.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.mlops.tracking import MLflowTracker
from src.mlops.evaluation import RAGEvaluator
from src.mlops.dvc_manager import DVCManager


class TestMLflowTracker:
    """Tests for MLflow wrapper."""
    
    @patch("src.mlops.tracking.mlflow")
    def test_tracker_initialization(self, mock_mlflow):
        """Test tracker setup."""
        tracker = MLflowTracker(tracking_uri="http://localhost:5000", experiment_name="test")
        
        mock_mlflow.set_tracking_uri.assert_called_with("http://localhost:5000")
        mock_mlflow.get_experiment_by_name.assert_called()
    
    @patch("src.mlops.tracking.mlflow")
    def test_log_metrics(self, mock_mlflow):
        """Test logging metrics."""
        tracker = MLflowTracker()
        tracker.log_metrics({"accuracy": 0.9})
        
        mock_mlflow.log_metrics.assert_called_with({"accuracy": 0.9}, step=None)


class TestRAGEvaluator:
    """Tests for RAGAS evaluator."""
    
    def test_evaluator_init(self):
        """Test evaluator init."""
        evaluator = RAGEvaluator()
        assert evaluator is not None
    
    @patch("src.mlops.evaluation.evaluate")
    @patch("src.mlops.evaluation.Dataset")
    def test_evaluation_flow(self, mock_dataset, mock_evaluate):
        """Test evaluation execution."""
        tracker = MagicMock()
        evaluator = RAGEvaluator(tracker=tracker)
        
        # Mock results
        mock_results = MagicMock()
        mock_results.to_pandas.return_value = MagicMock()
        mock_evaluate.return_value = mock_results
        
        scores = evaluator.evaluate(
            questions=["Q1"],
            answers=["A1"],
            contexts=[["C1"]],
        )
        
        mock_evaluate.assert_called()
        tracker.log_metrics.assert_called()


class TestDVCManager:
    """Tests for DVC manager."""
    
    @patch("src.mlops.dvc_manager.subprocess.run")
    def test_dvc_commands(self, mock_run):
        """Test dvc command execution."""
        dvc = DVCManager()
        
        # Test add
        dvc.add("data.csv")
        mock_run.assert_called()
        args = mock_run.call_args[0][0]
        assert args == ["dvc", "add", "data.csv"]
