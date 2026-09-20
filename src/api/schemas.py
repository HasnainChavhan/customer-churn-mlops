"""Pydantic v2 schemas for Customer Churn Prediction API."""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, field_validator


class CustomerFeatures(BaseModel):
    """Input features for a single customer churn prediction."""
    tenure: int = Field(..., ge=0, le=120, description="Months with the company", example=24)
    monthly_charges: float = Field(..., ge=0, le=500, description="Monthly charges in USD", example=65.5)
    total_charges: float = Field(..., ge=0, description="Total charges in USD", example=1572.0)
    contract_type: str = Field(..., description="Contract type", example="Month-to-month")
    internet_service: str = Field(..., description="Internet service type", example="Fiber optic")
    tech_support: str = Field(..., description="Has tech support", example="No")
    online_security: str = Field(..., description="Has online security", example="No")
    payment_method: str = Field(..., description="Payment method", example="Electronic check")
    senior_citizen: int = Field(..., ge=0, le=1, description="Is senior citizen (0/1)", example=0)
    partner: str = Field(..., description="Has partner (Yes/No)", example="Yes")
    dependents: str = Field(..., description="Has dependents (Yes/No)", example="No")

    @field_validator("contract_type")
    @classmethod
    def validate_contract(cls, v):
        valid = ["Month-to-month", "One year", "Two year"]
        if v not in valid:
            raise ValueError(f"contract_type must be one of {valid}")
        return v

    @field_validator("internet_service")
    @classmethod
    def validate_internet(cls, v):
        valid = ["DSL", "Fiber optic", "No"]
        if v not in valid:
            raise ValueError(f"internet_service must be one of {valid}")
        return v

    model_config = {"json_schema_extra": {"example": {
        "tenure": 24, "monthly_charges": 65.5, "total_charges": 1572.0,
        "contract_type": "Month-to-month", "internet_service": "Fiber optic",
        "tech_support": "No", "online_security": "No",
        "payment_method": "Electronic check", "senior_citizen": 0,
        "partner": "Yes", "dependents": "No",
    }}}


class PredictionResponse(BaseModel):
    """Churn prediction result for a single customer."""
    customer_id: str = Field(..., description="Auto-generated customer identifier")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn (0-1)")
    churn_prediction: bool = Field(..., description="True if predicted to churn")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, or HIGH")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    model_version: str = Field(..., description="Model version used for prediction")
    processing_time_ms: float = Field(default=0.0, description="Inference time in milliseconds")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Model metadata and performance metrics."""
    model_name: str
    model_version: str
    framework: str
    features: List[str]
    metrics: Dict[str, float]
    training_samples: int
    description: str
