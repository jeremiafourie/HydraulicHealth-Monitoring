"""
Module: modeling.py

Provides functions to load feature data, perform cross-validated training,
compute metrics, plot confusion matrices, and measure inference speed
for hydraulic system condition monitoring.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, f1_score, confusion_matrix

# Constants
target_columns = ["cooler_pct", "valve_pct", "pump_leak", "acc_pressure"]
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "f1_macro": make_scorer(f1_score, average="macro")
}


def load_features(path: str):
    """
    Load features CSV and split into X matrix and target arrays.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    targets : dict mapping target name to np.ndarray of shape (n_samples,)
    """
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in target_columns]
    X = df[feature_cols].values
    targets = {t: df[t].values for t in target_columns}
    return X, targets


def cross_validate_targets(X: np.ndarray, targets: dict, n_splits=5, random_state=42):
    """
    Perform stratified k-fold cross-validation for each target using a RandomForest.

    Returns
    -------
    results : dict mapping target name to CV results dict with accuracy & f1 arrays
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}
    for name, y in targets.items():
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        cv_res = cross_validate(clf, X, y, cv=cv, scoring=scoring)
        results[name] = cv_res
    return results


def summarize_cv_results(results: dict):
    """
    Compute mean & std for accuracy and f1_macro from cross-validation results.

    Returns
    -------
    summary : dict mapping target to dict with 'accuracy' and 'f1_macro' tuples
    """
    summary = {}
    for name, cv_res in results.items():
        acc_mean = cv_res['test_accuracy'].mean()
        acc_std  = cv_res['test_accuracy'].std()
        f1_mean  = cv_res['test_f1_macro'].mean()
        f1_std   = cv_res['test_f1_macro'].std()
        summary[name] = {
            'accuracy': (acc_mean, acc_std),
            'f1_macro': (f1_mean, f1_std)
        }
    return summary


def train_test_confusion(X: np.ndarray, y: np.ndarray, test_size=0.2, random_state=42, **clf_kwargs):
    """
    Train RandomForest on a single split and return confusion matrix.

    Parameters
    ----------
    X, y : data and labels
    test_size : float
    clf_kwargs : passed to RandomForestClassifier

    Returns
    -------
    cm : np.ndarray (n_classes, n_classes)
    labels : np.ndarray of class labels
    classifier : fitted classifier
    X_test, y_test : test data
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1, **clf_kwargs)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    labels = np.unique(y)
    cm = confusion_matrix(y_te, preds, labels=labels)
    return cm, labels, clf, X_te, y_te


def plot_confusion_matrix(cm: np.ndarray, labels: np.ndarray, title: str=None):
    """
    Plot a confusion matrix with seaborn.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    if title:
        ax.set_title(title)
    plt.show()


def measure_inference_speed(clf, X_test: np.ndarray, n_warmup=100):
    """
    Measure inference time per sample (ms) after a warm-up.

    Returns
    -------
    ms_per_sample : float
    """
    # Warm-up
    _ = clf.predict(X_test[:n_warmup])
    start = time.time()
    _ = clf.predict(X_test)
    end = time.time()
    total_ms = (end - start) * 1000
    return total_ms / len(X_test)


if __name__ == '__main__':
    # Example usage
    X, targets = load_features(os.path.join('..','data','processed','features.csv'))
    cv_results = cross_validate_targets(X, targets)
    summary = summarize_cv_results(cv_results)
    print('CV Summary:', summary)

    # Confusion matrices
    for name, y in targets.items():
        cm, labels, clf, X_te, y_te = train_test_confusion(X, y)
        plot_confusion_matrix(cm, labels, title=f'{name} Confusion Matrix')

    # Inference speed
    print('Inference speed (ms/sample):')
    for name, y in targets.items():
        _, _, clf, X_te, _ = train_test_confusion(X, y)
        speed = measure_inference_speed(clf, X_te)
        print(f'  {name}: {speed:.3f} ms/sample')
