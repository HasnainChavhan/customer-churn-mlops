"""pytest tests for the Customer Churn Prediction API."""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from fastapi.testclient import TestClient


@pytest.fixture
def mock_model():
    """Mock model that returns churn probability."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


@pytest.fixture
def mock_pipeline():
    """Mock preprocessing pipeline."""
    pipeline = MagicMock()
    pipeline.transform.return_value = np.zeros((1, 20))
    return pipeline


@pytest.fixture
def client(mock_model, mock_pipeline):
    """Test client with mocked model artifacts."""
    with patch.dict("src.api.main.model_artifacts", {
        "model": mock_model,
        "pipeline": mock_pipeline,
    }):
        from src.api.main import app
        yield TestClient(app)


VALID_CUSTOMER = {
    "tenure": 24,
    "monthly_charges": 65.5,
    "total_charges": 1572.0,
    "contract_type": "Month-to-month",
    "internet_service": "Fiber optic",
    "tech_support": "No",
    "online_security": "No",
    "payment_method": "Electronic check",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
}


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "metrics" in data
    assert "auc_roc" in data["metrics"]


def test_predict_single_valid(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_invalid_contract(client):
    bad_customer = VALID_CUSTOMER.copy()
    bad_customer["contract_type"] = "Invalid"
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422  # Validation error


def test_predict_batch(client):
    request = {"customers": [VALID_CUSTOMER, VALID_CUSTOMER]}
    response = client.post("/predict/batch", json=request)
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2
    assert "high_risk_count" in data


def test_predict_batch_too_large(client):
    request = {"customers": [VALID_CUSTOMER] * 1001}
    response = client.post("/predict/batch", json=request)
    assert response.status_code == 400
