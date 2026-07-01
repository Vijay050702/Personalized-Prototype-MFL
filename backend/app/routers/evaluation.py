from fastapi import APIRouter

from app.schemas.evaluation import EvaluationResponse, EvaluationSummary

router = APIRouter(tags=["Evaluation"])


@router.get("/evaluation", response_model=EvaluationSummary)
def get_evaluation():
    data = EvaluationResponse(
        accuracy=0.8734,
        precision=0.8654,
        recall=0.8812,
        f1_score=0.8732,
        auc_roc=0.9245,
        client_id="global",
        round=47,
        samples_evaluated=10000,
    )
    return EvaluationSummary(
        status="success", message="Evaluation results retrieved", data=data
    )
