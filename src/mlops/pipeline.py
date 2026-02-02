"""
MLOps Pipeline Orchestration.

Simple orchestration for:
- Data ingestion
- Indexing
- Evaluation runs
"""

import asyncio
import logging
from datetime import datetime

from src.mlops.tracking import create_tracker
from src.mlops.evaluation import RAGEvaluator
from src.mlops.dvc_manager import DVCManager

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates MLOps pipelines.
    """
    
    def __init__(self):
        self.tracker = create_tracker()
        self.dvc = DVCManager()
        self.evaluator = RAGEvaluator(tracker=self.tracker)
    
    async def run_evaluation_pipeline(
        self,
        test_dataset_path: str,
        n_samples: int = 50,
    ):
        """
        Run end-to-end evaluation pipeline.
        
        1. Load test dataset
        2. Run RAG pipeline on questions
        3. Collect results
        4. Run RAGAS evaluation
        5. Log to MLflow
        """
        from src.agents.workflow import create_rag_workflow
        import pandas as pd
        import json
        
        logger.info("Starting evaluation pipeline...")
        
        try:
            # 1. Load dataset
            with open(test_dataset_path, 'r') as f:
                dataset = json.load(f)
            
            # Limit samples
            samples = dataset[:n_samples]
            questions = [s["question"] for s in samples]
            ground_truths = [s.get("ground_truth", "") for s in samples]
            ground_truths_list = [[gt] for gt in ground_truths]  # RAGAS expects list of lists
            
            # 2. Run RAG
            workflow = create_rag_workflow()
            answers = []
            contexts = []
            
            logger.info("Running RAG generation...")
            for q in questions:
                state = await workflow.run(query=q)
                answers.append(state.get("final_answer", ""))
                
                # Extract context strings
                ctx_texts = [
                    c.get("content", "") 
                    for c in state.get("retrieved_contexts", [])
                ]
                contexts.append(ctx_texts)
            
            # 3. Evaluate
            logger.info("Running metrics evaluation...")
            scores = self.evaluator.evaluate(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=ground_truths_list,
                run_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            
            logger.info("Pipeline completed successfully.")
            return scores
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def run_data_versioning(self, data_path: str):
        """
        Version dataset using DVC.
        """
        logger.info(f"Versioning data at {data_path}...")
        if self.dvc.add(data_path):
            logger.info("Data added to DVC.")
        else:
            logger.error("Failed to add data to DVC.")


__all__ = ["PipelineOrchestrator"]
