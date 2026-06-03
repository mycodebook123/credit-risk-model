import pandas as pd
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_name: str = "credit-risk-best-model", version: int = 1):
    model_uri = f"models:/{model_name}/{version}"
    logger.info(f"Loading model from {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    return model


def predict_risk(model, input_data: dict) -> dict:
    df = pd.DataFrame([input_data])
    probability = model.predict_proba(df)[:, 1][0]
    prediction = int(probability >= 0.5)
    return {
        "is_high_risk": prediction,
        "risk_probability": round(float(probability), 4)
    }