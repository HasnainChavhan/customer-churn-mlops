"""Churn prediction model inference with SHAP explainability."""
import logging
from pathlib import Path
from typing import Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Loads trained model and pipeline for inference with SHAP explanation."""

    def __init__(self, model_path: str = "models/churn_model.pkl",
                 pipeline_path: str = "models/feature_pipeline.pkl"):
        self.model_path = Path(model_path)
        self.pipeline_path = Path(pipeline_path)
        self.model = None
        self.pipeline = None
        self.explainer = None

    def load(self):
        """Load model and pipeline from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}. Run training first.")
        self.model = joblib.load(self.model_path)
        self.pipeline = joblib.load(self.pipeline_path)
        self.explainer = shap.TreeExplainer(self.model)
        logger.info(f"Model loaded from {self.model_path}")

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on raw customer data.
        Returns: (predictions, probabilities)
        """
        X = self.pipeline.transform(data)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        return predictions, probabilities

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Return churn probabilities only."""
        X = self.pipeline.transform(data)
        return self.model.predict_proba(X)[:, 1]

    def explain(self, data: pd.DataFrame) -> dict:
        """
        Compute SHAP values for explainability.
        Returns feature importances for the given input.
        """
        if self.explainer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        X = self.pipeline.transform(data)
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        try:
            feature_names = self.pipeline.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        importance = dict(zip(feature_names, np.abs(shap_values[0]).tolist()))
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return {
            "shap_values": shap_values[0].tolist(),
            "feature_names": list(feature_names),
            "feature_importance": sorted_importance,
            "top_factors": list(sorted_importance.keys())[:5],
        }


if __name__ == "__main__":
    predictor = ChurnPredictor()
    predictor.load()
    sample = pd.DataFrame([{
        "tenure": 6, "monthly_charges": 89.0, "total_charges": 534.0,
        "contract_type": "Month-to-month", "internet_service": "Fiber optic",
        "tech_support": "No", "online_security": "No",
        "payment_method": "Electronic check", "senior_citizen": 0,
        "partner": "No", "dependents": "No",
    }])
    preds, probs = predictor.predict(sample)
    print(f"Prediction: {preds[0]}, Probability: {probs[0]:.4f}")
    explanation = predictor.explain(sample)
    print(f"Top factors: {explanation['top_factors']}")
