"""
MLflow Experiment Tracking.

Provides a wrapper around MLflow for tracking:
- Experiments
- Runs
- Parameters
- Metrics
- Artifacts
"""

import logging
import os
import tempfile
from contextlib import contextmanager
from typing import Any

import mlflow
from mlflow.entities import Run

from src.config import get_settings

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Wrapper for MLflow experiment tracking."""
    
    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "rag_evaluation",
    ):
        """
        Initialize tracker.
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Name of the experiment
        """
        settings = get_settings()
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self.experiment_name = experiment_name
        
        # Setup MLflow
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
            logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")
        
        self._setup_experiment()
    
    def _setup_experiment(self):
        """Create or set the experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"Failed to setup MLflow experiment: {e}")
    
    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
    ):
        """
        Context manager for an MLflow run.
        
        Args:
            run_name: Name for this run
            nested: Whether execution is nested in parent run
            tags: Tags to add to the run
        """
        try:
            run = mlflow.start_run(run_name=run_name, nested=nested, tags=tags)
            logger.info(f"Started MLflow run: {run.info.run_id}")
            yield run
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            yield None
        finally:
            mlflow.end_run()
    
    def log_params(self, params: dict[str, Any]):
        """Log parameters to current run."""
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"Failed to log params: {e}")
    
    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """Log metrics to current run."""
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        """Log a local file as an artifact."""
        try:
            if os.path.exists(local_path):
                mlflow.log_artifact(local_path, artifact_path)
            else:
                logger.warning(f"Artifact not found: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")
    
    def log_dict(self, dictionary: dict, artifact_file: str):
        """Log a dictionary as a JSON artifact."""
        try:
            mlflow.log_dict(dictionary, artifact_file)
        except Exception as e:
            logger.warning(f"Failed to log dictionary: {e}")
    
    def autolog(self):
        """Enable autologging where supported."""
        mlflow.autolog()


_tracker: MLflowTracker | None = None


def create_tracker(experiment_name: str = "rag_evaluation") -> MLflowTracker:
    """Factory to create or get tracker."""
    global _tracker
    if _tracker is None or _tracker.experiment_name != experiment_name:
        _tracker = MLflowTracker(experiment_name=experiment_name)
    return _tracker
