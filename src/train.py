import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_processed_data(filepath: str):
    df = pd.read_csv(filepath)
    feature_cols = [c for c in df.columns if c not in ['CustomerId', 'is_high_risk']]
    X = df[feature_cols]
    y = df['is_high_risk']
    return X, y


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }


def train_logistic_regression(X_train, y_train):
    logger.info("Training Logistic Regression")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, max_iter=1000))
    ])
    param_grid = {'model__C': [0.01, 0.1, 1.0, 10.0]}
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    search.fit(X_train, y_train)
    logger.info(f"Best LR params: {search.best_params_}")
    return search.best_estimator_


def train_random_forest(X_train, y_train):
    logger.info("Training Random Forest")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [5, 10, None]
    }
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    search.fit(X_train, y_train)
    logger.info(f"Best RF params: {search.best_params_}")
    return search.best_estimator_


def run_training(data_filepath: str):
    mlflow.set_experiment("credit-risk-model")

    X, y = load_processed_data(data_filepath)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    models = {
        'logistic_regression': train_logistic_regression,
        'random_forest': train_random_forest,
    }

    best_roc_auc = 0
    best_model = None
    best_model_name = None

    for name, train_fn in models.items():
        with mlflow.start_run(run_name=name):
            model = train_fn(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)

            # Log to MLflow
            mlflow.log_params(model.named_steps['model'].get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            logger.info(f"{name} metrics: {metrics}")

            if metrics['roc_auc'] > best_roc_auc:
                best_roc_auc = metrics['roc_auc']
                best_model = model
                best_model_name = name

    # Register best model
    logger.info(f"Best model: {best_model_name} with ROC-AUC: {best_roc_auc:.4f}")
    with mlflow.start_run(run_name=f"best_{best_model_name}"):
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            registered_model_name="credit-risk-best-model"
        )
        mlflow.log_metric("roc_auc", best_roc_auc)
        logger.info("Best model registered in MLflow Model Registry")

    return best_model


if __name__ == "__main__":
    run_training("data/processed/processed_data.csv")