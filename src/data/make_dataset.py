import os
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

def main():
    # Load config
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {"data": {"raw_path": "data/raw/churn_data.csv", "random_state": 42}}

    np.random.seed(config["data"].get("random_state", 42))
    
    n_samples = 1000
    
    data = {
        "tenure": np.random.randint(1, 72, n_samples),
        "monthly_charges": np.round(np.random.uniform(20.0, 120.0, n_samples), 2),
        "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], n_samples),
        "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], n_samples),
        "tech_support": np.random.choice(["Yes", "No", "No internet service"], n_samples)
    }
    
    df = pd.DataFrame(data)
    df["total_charges"] = np.round(df["tenure"] * df["monthly_charges"] * np.random.uniform(0.9, 1.1, n_samples), 2)
    
    # Generate target based on features (higher churn for month-to-month, short tenure)
    churn_prob = np.zeros(n_samples)
    churn_prob += np.where(df["contract_type"] == "Month-to-month", 0.4, 0.0)
    churn_prob += np.where(df["tenure"] < 12, 0.3, 0.0)
    churn_prob += np.where(df["monthly_charges"] > 80, 0.2, 0.0)
    churn_prob += np.where(df["tech_support"] == "No", 0.1, 0.0)
    
    churn_prob = np.clip(churn_prob, 0, 1)
    df["churn"] = np.random.binomial(1, churn_prob)
    
    output_path = Path(config["data"]["raw_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved to {output_path}")

if __name__ == "__main__":
    main()
