"""
Module: modeling.py

Provides generic functions for loading feature data, cross-validation,
train/test evaluation, plotting confusion matrices, measuring inference speed,
and saving/loading/publishing models.
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, make_scorer
from sklearn.base import clone
from typing import Dict, Any

# Default target column names
TARGET_COLUMNS = ["cooler_pct", "valve_pct", "pump_leak", "acc_pressure"]

# Default cross-validation splitter and scoring metrics
DEFAULT_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
DEFAULT_SCORING = {
    "accuracy": make_scorer(accuracy_score),
    "f1_macro": make_scorer(f1_score, average="macro")
}


def load_features(path: str):
    """
    Load features CSV and return feature matrix X, target dict, and feature column names.
    """
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
    X = df[feature_cols].values
    targets = {t: df[t].values for t in TARGET_COLUMNS}
    return X, targets, feature_cols


def cross_validate_model(estimator: Any, X: np.ndarray, y: np.ndarray,
                         cv=None, scoring=None) -> dict:
    """
    Cross-validate an estimator on (X, y) and return the cross_validate output.
    """
    cv = cv or DEFAULT_CV
    scoring = scoring or DEFAULT_SCORING
    return cross_validate(clone(estimator), X, y, cv=cv, scoring=scoring)


def summarize_cv_results(cv_res: dict) -> dict:
    """
    Summarize cross-validation results: mean and std for each scoring metric.
    Returns {metric: (mean, std)}.
    """
    summary = {}
    for key, vals in cv_res.items():
        if key.startswith("test_"):
            metric = key.replace("test_", "")
            summary[metric] = (np.mean(vals), np.std(vals))
    return summary


def train_test_evaluate(estimator: Any, X: np.ndarray, y: np.ndarray,
                        test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Perform train/test split, fit estimator, and compute metrics and confusion matrix.
    Returns a dict with:
      - 'model': fitted estimator
      - 'X_test', 'y_test', 'y_pred'
      - 'metrics': {'accuracy': ..., 'f1_macro': ...}
      - 'confusion_matrix', 'labels'
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
    labels = np.unique(y_te)
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    return {
        'model': model,
        'X_test': X_te,
        'y_test': y_te,
        'y_pred': y_pred,
        'metrics': metrics,
        'confusion_matrix': cm,
        'labels': labels
    }


def plot_confusion_matrix(cm: np.ndarray, labels: np.ndarray,
                          title: str = None, figsize=(6, 4)):
    """
    Plot a confusion matrix using seaborn heatmap.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    if title:
        ax.set_title(title)
    plt.show()


def measure_inference_speed(model: Any, X_test: np.ndarray, n_warmup: int = 100) -> float:
    """
    Measure inference time per sample (ms) after a warm-up phase.
    """
    _ = model.predict(X_test[:n_warmup])
    start = time.time()
    _ = model.predict(X_test)
    end = time.time()
    total_ms = (end - start) * 1000
    return total_ms / len(X_test)


def save_model(model: Any, path: str) -> None:
    """
    Save a trained model to disk using joblib.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    """
    Load and return a model saved with joblib.
    """
    return joblib.load(path)


def publish_models(models: Dict[str, Any], output_dir: str) -> None:
    """
    Save multiple models provided as a dict of name->model to output_dir.
    Filenames will be '<name>.pkl'.

    Parameters
    ----------
    models : dict
        Mapping from model name (e.g. 'rf_cooler') to trained estimator.
    output_dir : str
        Directory to save model files into.
    """
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        fname = f"{name}.pkl"
        path = os.path.join(output_dir, fname)
        save_model(model, path)


if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier

    # Example: train and publish RF models for each target
    base = os.path.dirname(__file__)
    FEATURES_CSV = os.path.join(base, '..','data','processed','features.csv')
    X, targets, _ = load_features(FEATURES_CSV)
    models_to_publish = {}
    for tgt, y in targets.items():
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        out = train_test_evaluate(rf, X, y)
        models_to_publish[f"rf_{tgt}"] = out['model']
    publish_models(models_to_publish, os.path.join('..','artifacts'))
