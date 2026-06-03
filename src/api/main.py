from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
from src.predict import load_model, predict_risk
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts credit risk probability for Bati Bank's buy-now-pay-later service",
    version="1.0.0"
)

model = None


@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/")
def root():
    return {"message": "Credit Risk Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = predict_risk(model, features.model_dump())
        logger.info(f"Prediction made: {result}")
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))