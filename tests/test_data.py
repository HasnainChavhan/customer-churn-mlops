"""pytest tests for dataset generation."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.make_dataset import generate_churn_dataset


def test_dataset_generation():
    df = generate_churn_dataset(n_samples=500, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500


def test_dataset_columns():
    df = generate_churn_dataset(n_samples=100, random_state=42)
    required_cols = [
        "tenure", "monthly_charges", "total_charges", "contract_type",
        "internet_service", "tech_support", "online_security",
        "payment_method", "senior_citizen", "partner", "dependents", "churn"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_dataset_target_distribution():
    df = generate_churn_dataset(n_samples=5000, random_state=42)
    churn_rate = df["churn"].mean()
    assert 0.1 <= churn_rate <= 0.5, f"Churn rate {churn_rate} is unrealistic"


def test_dataset_no_nulls_in_numeric():
    df = generate_churn_dataset(n_samples=500, random_state=42)
    assert df["tenure"].isna().sum() == 0
    assert df["monthly_charges"].isna().sum() == 0


def test_dataset_value_ranges():
    df = generate_churn_dataset(n_samples=500, random_state=42)
    assert (df["tenure"] >= 0).all()
    assert (df["monthly_charges"] > 0).all()
    assert (df["senior_citizen"].isin([0, 1])).all()
