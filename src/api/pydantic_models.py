from pydantic import BaseModel


class CustomerFeatures(BaseModel):
    total_transaction_amount: float
    avg_transaction_amount: float
    transaction_count: int
    std_transaction_amount: float
    total_value: float
    avg_value: float
    avg_hour: float
    avg_day: float
    avg_month: float
    recency: float
    frequency: float
    monetary: float


class PredictionResponse(BaseModel):
    is_high_risk: int
    risk_probability: float