import yaml
import joblib
import pandas as pd
import mlflow
import optuna
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

def train():
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Note: Ensure data and models directories are created. Assuming build_features saved them in data/processed
    processed_dir = Path(config["data"]["processed_path"]).parent
    X_train = pd.read_csv(processed_dir / "X_train.csv").values
    y_train = pd.read_csv(processed_dir / "y_train.csv").values.ravel()
    X_test = pd.read_csv(processed_dir / "X_test.csv").values
    y_test = pd.read_csv(processed_dir / "y_test.csv").values.ravel()

    # SMOTE for class imbalance
    smote = SMOTE(random_state=config["data"]["random_state"])
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    mlflow.set_tracking_uri(config["model"].get("mlflow_tracking_uri", "http://localhost:5000"))
    mlflow.set_experiment(config["model"]["mlflow_experiment_name"])

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "random_state": config["data"]["random_state"]
        }
        model = XGBClassifier(**params)
        model.fit(X_train_res, y_train_res)
        preds = model.predict(X_test)
        return f1_score(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config["model"]["n_trials"])

    best_params = study.best_params
    best_params["random_state"] = config["data"]["random_state"]

    with mlflow.start_run():
        model = XGBClassifier(**best_params)
        model.fit(X_train_res, y_train_res)
        
        preds = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "auc_roc": roc_auc_score(y_test, pred_proba),
            "f1": f1_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds)
        }
        
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")
        
        model_path = Path(config["model"]["path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}. F1 Score: {metrics['f1']:.4f}")

if __name__ == "__main__":
    train()
