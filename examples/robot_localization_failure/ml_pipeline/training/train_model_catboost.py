# ml_pipeline/training/train_model_catboost.py

import argparse
from pathlib import Path
import polars as pl
import numpy as np
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

from ml_pipeline.utils.paths import DATASETS
from ml_pipeline.utils.common import (
    load_dataset,
    prepare_features,
    apply_scaler,
    fit_scaler,
    compute_metrics,
    print_metrics,
    save_model,
)

MODEL_NAME = "catboost"


def main():
    # --------------------------------------------------------
    # Parse arguments
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train CatBoost model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="train",
        help="Dataset name inside artifacts/datasets/, e.g. train or train_temporal",
    )
    args = parser.parse_args()

    dataset_path = DATASETS / f"{args.dataset}.parquet"
    if not dataset_path.exists():
        print("\n❌ Dataset not found:", dataset_path)
        print("Available datasets:")
        for f in DATASETS.iterdir():
            if f.is_file():
                print(" -", f.name)
        return

    print(f"\n📦 Using dataset: {dataset_path}")

    # --------------------------------------------------------
    # Load & prepare dataset
    # --------------------------------------------------------
    df = load_dataset(dataset_path)
    X, y, feature_cols = prepare_features(df)

    print("Splitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=True,
        random_state=42
    )

    # --------------------------------------------------------
    # CatBoost does ITS OWN feature scaling internally → no scaler
    # --------------------------------------------------------
    scaler = fit_scaler(X_train, model_type="catboost")
    X_train_scaled = apply_scaler(X_train, scaler)
    X_val_scaled = apply_scaler(X_val, scaler)

    # --------------------------------------------------------
    # Train CatBoost
    # --------------------------------------------------------
    print("\n🚀 Training CatBoost...")

    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=8,
        loss_function="Logloss",
        eval_metric="F1",
        class_weights=[1.0, 2.5],   # give rare class more weight
        random_seed=42,
        verbose=200,
        task_type="CPU",
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=(X_val_scaled, y_val),
        use_best_model=True,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    y_pred = model.predict(X_val_scaled).astype(int)

    print("\n=== Validation Performance ===")
    metrics = compute_metrics(y_val, y_pred)
    print_metrics(metrics)

    # --------------------------------------------------------
    # Save model package
    # --------------------------------------------------------
    save_model(
        model_name=f"{MODEL_NAME}_{args.dataset}",
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        metrics=metrics
    )


if __name__ == "__main__":
    main()
