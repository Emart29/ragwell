"""API routes for evaluation."""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

from evaluation.run_eval import run_evaluation


router = APIRouter()


class EvaluationResponse(BaseModel):
    """Evaluation response."""
    results: Dict[str, Any]
    message: str


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_api():
    """Run evaluation harness via API.
    
    Returns:
        Comparison table with all metrics
    """
    results = run_evaluation()
    
    return EvaluationResponse(
        results=results,
        message="Evaluation complete. See results for detailed metrics."
    )
