import argparse
import json
import joblib
import numpy as np
import polars as pl
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from ml_pipeline.utils.paths import DATASETS, MODELS
from ml_pipeline.utils.common import (
    load_dataset,
    prepare_features,
    fit_scaler,
    apply_scaler,
    compute_metrics,
    print_metrics,
    save_model,
)


MODEL_NAME = "xgboost"


# ------------------------------------------------------
# Argument parsing + usage instructions
# ------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train XGBoost model on AMCL quality dataset"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Which dataset to use: 'train' or 'train_temporal'"
    )

    args = parser.parse_args()

    # Validate
    valid = ["train", "train_temporal"]
    if args.dataset not in valid:
        print("\n❌ ERROR: You must specify a valid --dataset argument.\n")
        print("Usage:")
        print("  python3 -m ml_pipeline.training.train_model_xgb --dataset train")
        print("  python3 -m ml_pipeline.training.train_model_xgb --dataset train_temporal\n")
        print("Available options:")
        print("  train            → artifacts/datasets/train.parquet")
        print("  train_temporal   → artifacts/datasets/train_temporal.parquet\n")
        exit(1)

    return args


def main():

    args = parse_args()   # <-- validated args

    # ============================================================
    # Dataset selection
    # ============================================================
    dataset_file = (
        DATASETS / "train.parquet"
        if args.dataset == "train"
        else DATASETS / "train_temporal.parquet"
    )

    print(f"\n📦 Using dataset: {dataset_file}")

    df = load_dataset(dataset_file)
    X, y, feature_cols = prepare_features(df)

    # ============================================================
    # Train/val split
    # ============================================================
    print("\nSplitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=True,
        random_state=42
    )

    # ============================================================
    # Scaling: XGB does NOT require scaling → scaler = None
    # ============================================================
    scaler = fit_scaler(X_train, model_type="xgb")

    X_train_scaled = apply_scaler(X_train, scaler)
    X_val_scaled = apply_scaler(X_val, scaler)

    # ============================================================
    # Define model
    # ============================================================
    print(f"\n🚀 Training {MODEL_NAME}...")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )

    model.fit(X_train_scaled, y_train)

    # ============================================================
    # Evaluate
    # ============================================================
    y_pred = model.predict(X_val_scaled)

    print("\n=== Validation Performance ===")
    metrics = compute_metrics(y_val, y_pred)
    print_metrics(metrics)

    # ============================================================
    # Save model package
    # ============================================================
    save_model(
        model_name=f"{MODEL_NAME}_{args.dataset}",
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        metrics=metrics
    )


if __name__ == "__main__":
    main()
