import pandas as pd
import yaml
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def main():
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "data": {"raw_path": "data/raw/churn_data.csv", "processed_path": "data/processed/churn_data_processed.csv", "test_size": 0.2, "random_state": 42},
            "features": {"numerical_cols": ["tenure", "monthly_charges", "total_charges"], "categorical_cols": ["contract_type", "internet_service", "tech_support"], "pipeline_path": "models/feature_pipeline.joblib"}
        }

    df = pd.read_csv(config["data"]["raw_path"])
    
    num_cols = config["features"]["numerical_cols"]
    cat_cols = config["features"]["categorical_cols"]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])

    X = df.drop("churn", axis=1)
    y = df["churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"]
    )
    
    preprocessor.fit(X_train)
    
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save processed data (optional, useful for debugging)
    processed_dir = Path(config["data"]["processed_path"]).parent
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_train_processed).to_csv(processed_dir / "X_train.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    pd.DataFrame(X_test_processed).to_csv(processed_dir / "X_test.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)
    
    # Save pipeline
    pipeline_path = Path(config["features"]["pipeline_path"])
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, pipeline_path)
    print(f"Feature pipeline saved to {pipeline_path}")

if __name__ == "__main__":
    main()
