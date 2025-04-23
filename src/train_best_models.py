#!/usr/bin/env python3
"""
train_best_models.py

For each target variable:
 1. Load precomputed features.
 2. Perform a randomized search over RandomForest hyperparameters (CV F1_macro).
 3. Retrain the best RandomForest on the full dataset.
 4. Save the fitted model to artifacts/rf_{target}.pkl and record best params.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.modeling import load_features

def main():
    # Paths
    project_root = Path(__file__).resolve().parent
    feature_csv   = project_root / "data" / "processed" / "features.csv"
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Load data
    X, targets, feature_names = load_features(str(feature_csv))
    print(f"Loaded X: {X.shape}, targets: {list(targets.keys())}")

    # Cross-validation setup
    CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Hyperparameter search space
    param_dist = {
        "n_estimators":       [50, 100, 200, 300],
        "max_depth":          [None, 10, 20, 30],
        "min_samples_split":  [2, 5, 10],
        "min_samples_leaf":   [1, 2, 4],
        "bootstrap":          [True, False],
    }

    best_params = {}

    for tgt, y in targets.items():
        print(f"\n=== Tuning RandomForest for target '{tgt}' ===")
        rf = RandomForestClassifier(random_state=42)

        rs = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=30,
            scoring="f1_macro",
            cv=CV,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
        rs.fit(X, y)

        # Record best hyperparameters
        best_params[tgt] = {
            "params": rs.best_params_,
            "cv_f1_macro": rs.best_score_,
        }
        print(f"→ Best CV F1 macro: {rs.best_score_:.4f}")
        print(f"→ Params: {rs.best_params_}")

        # Retrain on full data
        best_rf = RandomForestClassifier(random_state=42, **rs.best_params_)
        best_rf.fit(X, y)

        # Save the model with rf_{target}.pkl naming
        model_filename = f"rf_{tgt}.pkl"
        model_path = artifacts_dir / model_filename
        joblib.dump(best_rf, model_path)
        print(f"Saved model to {model_path}")

    # Write out all best‐params for reference
    params_path = artifacts_dir / "best_random_forest_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved all best‐params to {params_path}\n")


if __name__ == "__main__":
    main()
