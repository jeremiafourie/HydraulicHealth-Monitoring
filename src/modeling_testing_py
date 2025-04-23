"""
Module: modeling.py

Functions for loading features, training and evaluating multiple models across different
hydraulic system targets (cooler_pct, valve_pct, pump_leak, acc_pressure).
Supports cross-validation, train/test evaluation, classification reports, and grouped comparisons.
"""
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

# Default target columns
TARGET_COLUMNS = ["cooler_pct", "valve_pct", "pump_leak", "acc_pressure"]
# Cross-validation splitter and scoring metrics
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = {"accuracy": "accuracy", "f1_macro": "f1_macro"}


def load_features(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Load a feature CSV and return (X, targets_dict, feature_names).
    """
    df = pd.read_csv(path)
    feature_names = np.array([c for c in df.columns if c not in TARGET_COLUMNS])
    X = df[feature_names].values
    targets = {t: df[t].values for t in TARGET_COLUMNS}
    return X, targets, feature_names


def train_model(estimator: Any, X: np.ndarray, y: np.ndarray) -> Any:
    """
    Train an estimator on (X, y) and return the fitted model.
    """
    model = clone(estimator)
    model.fit(X, y)
    return model


def cross_validate_model(estimator: Any, X: np.ndarray, y: np.ndarray,
                         cv: StratifiedKFold = CV, scoring: dict = SCORING) -> pd.DataFrame:
    """
    Perform cross-validation and return a DataFrame of mean±std for each metric.
    """
    results = cross_validate(clone(estimator), X, y, cv=cv, scoring=scoring)
    summary = {}
    for key, vals in results.items():
        if key.startswith('test_'):
            metric = key.replace('test_', '')
            summary[f"{metric}_mean"] = np.mean(vals)
            summary[f"{metric}_std"]  = np.std(vals)
    return pd.DataFrame([summary])


def train_test_evaluate(estimator: Any, X: np.ndarray, y: np.ndarray,
                        test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Perform train/test split, fit estimator, and compute metrics and classification report.
    Returns dict with keys:
      - model: fitted estimator
      - metrics: {'accuracy', 'f1_macro'}
      - confusion_matrix: np.ndarray
      - report: str classification report
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    model = clone(estimator)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = {
        'accuracy': accuracy_score(y_te, y_pred),
        'f1_macro': f1_score(y_te, y_pred, average='macro')
    }
    cm = confusion_matrix(y_te, y_pred, labels=np.unique(y_te))
    report = classification_report(y_te, y_pred)
    return {
        'model': model,
        'metrics': metrics,
        'confusion_matrix': cm,
        'report': report
    }


def evaluate_models(models: Dict[str, Any], X: np.ndarray,
                    targets: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Train/test evaluate multiple models across all targets.
    Returns:
      - metrics_df: DataFrame with columns [model, target, accuracy, f1_macro]
      - reports: dict mapping 'model_target' to classification report string
    """
    records = []
    reports = {}
    for model_name, estimator in models.items():
        for target_name, y in targets.items():
            out = train_test_evaluate(estimator, X, y)
            rec = {
                'model': model_name,
                'target': target_name,
                'accuracy': out['metrics']['accuracy'],
                'f1_macro': out['metrics']['f1_macro']
            }
            records.append(rec)
            reports[f"{model_name}_{target_name}"] = out['report']
    metrics_df = pd.DataFrame(records)
    return metrics_df, reports


def save_model(model: Any, path: str) -> None:
    """Save a model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    """Load a model from disk."""
    return joblib.load(path)


def publish_models(models: Dict[str, Any], output_dir: str) -> None:
    """Save multiple models into output_dir as '<name>.pkl'."""
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        save_model(model, os.path.join(output_dir, f"{name}.pkl"))


if __name__ == '__main__':
    # Example usage: evaluate three candidate models for all targets
    X, targets, feature_names = load_features('data/processed/features.csv')
    candidates = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVC': SVC(random_state=42)
    }
    # Cross-validation summary per model/target
    print("Cross-Validation Summary:")
    for name, estimator in candidates.items():
        for target, y in targets.items():
            df_cv = cross_validate_model(estimator, X, y)
            print(f"{name} on {target}:")
            print(df_cv.to_string(index=False), '\n')
    # Train/test evaluation grouped
    metrics_df, reports = evaluate_models(candidates, X, targets)
    print("Grouped Train/Test Metrics:")
    print(metrics_df)
    # Detailed classification reports
    print("\nClassification Reports:")
    for key, rpt in reports.items():
        print(f"--- {key} ---")
        print(rpt)
    # Optionally save best models
    # best_rf = train_model(candidates['RandomForest'], X, targets['cooler_pct'])
    # save_model(best_rf, 'artifacts/best_rf_cooler_pct.pkl')
