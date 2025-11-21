# ml_pipeline/training/train_model_optuna_xgb.py

import optuna
import joblib
import json
import argparse
import polars as pl
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml_pipeline.utils.paths import DATASETS, MODELS
from ml_pipeline.utils.common import (
    load_dataset,
    prepare_features,
    apply_scaler,
    save_model,
)

MODEL_NAME = "xgboost_optuna"


# ============================================================
# Optuna objective
# ============================================================
def build_objective(X_train, X_val, y_train, y_val):

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "lambda": trial.suggest_float("lambda", 0.0, 5.0),
            "alpha": trial.suggest_float("alpha", 0.0, 5.0),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
        }

        model = XGBClassifier(**params)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        # F0.5 emphasizes precision → perfect for delocalization
        f05 = fbeta_score(y_val, y_pred, beta=0.5)

        return f05

    return objective


# ============================================================
# Main entry point
# ============================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="train.parquet",
        help="Choose dataset: train.parquet or train_temporal.parquet",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default 50)",
    )
    args = parser.parse_args()

    dataset_path = DATASETS / args.dataset
    if not dataset_path.exists():
        print("❌ Dataset not found:", dataset_path)
        print("Available datasets:")
        for f in DATASETS.iterdir():
            print(" -", f.name)
        return

    print(f"\n📦 Using dataset: {dataset_path}")

    df = load_dataset(dataset_path)
    X, y, feature_cols = prepare_features(df)

    # ========================================================
    # Split dataset
    # ========================================================
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    # XGBoost does NOT need scaling, but Optuna tries many parameter combos,
    # so we keep everything consistent
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # ========================================================
    # Optuna study
    # ========================================================
    print("\n🚀 Starting Optuna optimization (target: F0.5)")
    objective = build_objective(X_train_scaled, X_val_scaled, y_train, y_val)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print("\n=== Best Trial ===")
    print("Score (F0.5):", study.best_trial.value)
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"{k}: {v}")

    # ========================================================
    # Train best model on full split
    # ========================================================
    best_params = study.best_params
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
    })

    model = XGBClassifier(**best_params)
    model.fit(X_train_scaled, y_train)

    # Final validation
    y_pred = model.predict(X_val_scaled)
    f05 = fbeta_score(y_val, y_pred, beta=0.5)

    print("\nFinal F0.5 on validation:", f05)

    # ========================================================
    # Save model
    # ========================================================
    save_model(
        model_name=MODEL_NAME,
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        metrics={"F0.5": f05, "best_params": best_params},
    )

    # Save study for later analysis
    study_path = MODELS / f"{MODEL_NAME}_study.pkl"
    joblib.dump(study, study_path)

    print("✔ Saved study →", study_path)


if __name__ == "__main__":
    main()
