# 🔄 Customer Churn MLOps Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.8-orange?logo=mlflow)](https://mlflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://docker.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=github)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **End-to-end MLOps pipeline** for predicting customer churn using XGBoost with automated hyperparameter tuning (Optuna), experiment tracking (MLflow), explainability (SHAP), and a production-ready REST API (FastAPI).

---

## 📌 Business Problem

Telecom companies lose **$1,600+ per churned customer** on average. Predicting churn 30 days in advance enables targeted retention campaigns — this system identifies at-risk customers with **AUC-ROC > 0.88** and provides explainability for each prediction.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Customer Churn MLOps                       │
│                                                              │
│  Raw Data ──► Feature Pipeline ──► XGBoost Model            │
│     │               │                    │                   │
│     ▼               ▼                    ▼                   │
│  CSV/DB        sklearn Pipeline      MLflow Registry          │
│                (Scaler+OHE)          (Versioned)              │
│                     │                    │                   │
│                     └────────┬───────────┘                   │
│                              ▼                               │
│                        FastAPI Server                        │
│                      /predict  /batch                        │
│                      /health   /model/info                   │
│                              │                               │
│                     MLflow Tracking UI                       │
│                    (Experiments + Models)                    │
└──────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- 🤖 **XGBoost** with **Optuna** hyperparameter optimization (automated tuning)
- 📊 **MLflow** experiment tracking — log params, metrics, artifacts, and model
- ⚖️ **SMOTE** oversampling for class imbalance handling
- 🔍 **SHAP** values for prediction explainability
- 🚀 **FastAPI** REST API — single prediction & batch prediction
- 🐳 **Docker** + Docker Compose — one-command deployment
- ✅ **pytest** test suite with CI/CD via GitHub Actions
- 📈 Model metrics: Accuracy, AUC-ROC, F1, Precision, Recall

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost 2.0 |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow |
| Feature Engineering | scikit-learn Pipeline |
| Class Imbalance | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| API Framework | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |
| Testing | pytest + httpx |
| CI/CD | GitHub Actions |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/HasnainChavhan/customer-churn-mlops
cd customer-churn-mlops

# 2. Create virtual environment
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate dataset and train model
make train

# 5. Start API server (Docker)
docker-compose up --build
```

API will be available at: **http://localhost:8000**
MLflow UI at: **http://localhost:5000**

---

## 📡 API Documentation

### POST `/predict` — Single Prediction
```json
POST http://localhost:8000/predict
{
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
  "dependents": "No"
}
```
Response:
```json
{
  "customer_id": "cust_abc123",
  "churn_probability": 0.73,
  "churn_prediction": true,
  "risk_level": "HIGH",
  "confidence": 0.73,
  "model_version": "1.0.0"
}
```

### POST `/predict/batch` — Batch Prediction
Send array of customer objects, returns array of predictions.

### GET `/health` — Health Check
### GET `/model/info` — Model metadata and performance metrics

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| AUC-ROC | 0.88 |
| F1 Score | 0.79 |
| Precision | 0.82 |
| Recall | 0.76 |
| Accuracy | 0.84 |

---

## 📁 Project Structure

```
customer-churn-mlops/
├── .github/workflows/ci.yml    # CI/CD pipeline
├── config/config.yaml          # Configuration
├── src/
│   ├── data/make_dataset.py    # Synthetic dataset generation
│   ├── features/build_features.py  # Feature engineering pipeline
│   ├── models/
│   │   ├── train_model.py      # XGBoost + Optuna + MLflow
│   │   └── predict_model.py   # Inference + SHAP
│   └── api/
│       ├── main.py             # FastAPI application
│       └── schemas.py          # Pydantic models
├── tests/                      # pytest test suite
├── notebooks/                  # Exploratory analysis
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-service setup
├── requirements.txt
└── Makefile                    # Convenience commands
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) file.

---

*Built by [Hasnain Chavhan](https://github.com/HasnainChavhan) — Open to Data Science / MLOps roles*
