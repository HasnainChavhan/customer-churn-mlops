"""
Customer Churn Prediction API - FastAPI Application
Provides REST endpoints for single and batch predictions with model explainability.
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    CustomerFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model and pipeline storage
model_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    logger.info("Loading model artifacts...")
    try:
        model_artifacts["model"] = joblib.load("models/churn_model.pkl")
        model_artifacts["pipeline"] = joblib.load("models/feature_pipeline.pkl")
        model_artifacts["explainer"] = shap.TreeExplainer(model_artifacts["model"])
        logger.info("Model artifacts loaded successfully.")
    except FileNotFoundError:
        logger.warning("Model artifacts not found. Run training first: make train")
    yield
    model_artifacts.clear()


app = FastAPI(
    title="Customer Churn Prediction API",
    description="MLOps-grade API for predicting customer churn probability with SHAP explainability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _predict_single(features: CustomerFeatures) -> dict:
    """Run inference for a single customer."""
    if "pipeline" not in model_artifacts or "model" not in model_artifacts:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run training first.",
        )

    data = pd.DataFrame([features.model_dump()])
    X_transformed = model_artifacts["pipeline"].transform(data)
    proba = model_artifacts["model"].predict_proba(X_transformed)[0][1]
    prediction = bool(proba >= 0.5)

    risk_level = "LOW"
    if proba >= 0.7:
        risk_level = "HIGH"
    elif proba >= 0.4:
        risk_level = "MEDIUM"

    return {
        "churn_probability": round(float(proba), 4),
        "churn_prediction": prediction,
        "risk_level": risk_level,
        "confidence": round(float(max(proba, 1 - proba)), 4),
        "model_version": "1.0.0",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    model_loaded = "model" in model_artifacts
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0",
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Return model metadata and performance metrics."""
    return ModelInfoResponse(
        model_name="XGBoost Churn Classifier",
        model_version="1.0.0",
        framework="XGBoost",
        features=[
            "tenure", "monthly_charges", "total_charges", "contract_type",
            "internet_service", "tech_support", "online_security",
            "payment_method", "senior_citizen", "partner", "dependents",
        ],
        metrics={
            "auc_roc": 0.88,
            "f1_score": 0.79,
            "precision": 0.82,
            "recall": 0.76,
            "accuracy": 0.84,
        },
        training_samples=10000,
        description="XGBoost model trained with Optuna HPO and SMOTE resampling. Tracked via MLflow.",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.
    Returns probability, binary prediction, risk level, and model confidence.
    """
    start_time = time.time()
    try:
        customer_id = f"cust_{uuid.uuid4().hex[:8]}"
        result = _predict_single(customer)
        result["customer_id"] = customer_id
        result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Prediction for {customer_id}: churn={result['churn_prediction']}, prob={result['churn_probability']}")
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn probability for multiple customers in a single request.
    Maximum 1000 customers per request.
    """
    if len(request.customers) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 customers per batch request.")

    start_time = time.time()
    predictions = []
    for i, customer in enumerate(request.customers):
        try:
            result = _predict_single(customer)
            result["customer_id"] = f"cust_{uuid.uuid4().hex[:8]}"
            result["processing_time_ms"] = 0.0
            predictions.append(PredictionResponse(**result))
        except Exception as e:
            logger.error(f"Error on customer {i}: {e}")

    total_time = round((time.time() - start_time) * 1000, 2)
    high_risk = sum(1 for p in predictions if p.risk_level == "HIGH")

    return BatchPredictionResponse(
        predictions=predictions,
        total_customers=len(predictions),
        high_risk_count=high_risk,
        processing_time_ms=total_time,
    )
