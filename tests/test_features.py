import os
from pathlib import Path
import yaml
import pytest
from src.data.make_dataset import main as make_dataset_main
from src.features.build_features import main as build_features_main

def test_data_generation():
    # Set up config for testing if needed or just run it
    make_dataset_main()
    assert Path("data/raw/churn_data.csv").exists()

def test_feature_building():
    build_features_main()
    assert Path("models/feature_pipeline.joblib").exists()
    assert Path("data/processed/X_train.csv").exists()
