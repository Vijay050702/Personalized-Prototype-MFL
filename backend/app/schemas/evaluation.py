from pydantic import BaseModel


class EvaluationResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    client_id: str
    round: int
    samples_evaluated: int


class EvaluationSummary(BaseModel):
    status: str
    message: str
    data: EvaluationResponse
