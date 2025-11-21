# ml_pipeline/training/train_model_nn.py

import polars as pl
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from ml_pipeline.utils.paths import DATASETS
from ml_pipeline.utils.common import (
    load_dataset,
    prepare_features,
    fit_scaler,
    apply_scaler,
    compute_metrics,
    print_metrics,
    save_model,
)

MODEL_NAME = "nn"   # or "mlp", up to you


def main():
    # ============================================================
    # 1) Load & prepare dataset
    # ============================================================
    print("Loading training data...")
    df = load_dataset(DATASETS / "train.parquet")

    print("Preparing features...")
    X, y, feature_cols = prepare_features(df)

    # ============================================================
    # 2) Train/val split
    # ============================================================
    print("Splitting train/val...")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
        stratify=y,  # keep class balance similar in train/val
    )

    # ============================================================
    # 3) Scaling (NN *needs* scaling)
    # ============================================================
    scaler = fit_scaler(X_train, model_type="nn")  # returns StandardScaler

    X_train_scaled = apply_scaler(X_train, scaler)
    X_val_scaled = apply_scaler(X_val, scaler)

    # ============================================================
    # 4) Define & train neural network model
    # ============================================================
    print(f"\nTraining {MODEL_NAME} (MLPClassifier)...")

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,          # L2 regularization
        batch_size=128,
        learning_rate="adaptive",
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,  # internal early-stopping split
        random_state=42,
        verbose=False,
    )

    model.fit(X_train_scaled, y_train)

    # ============================================================
    # 5) Evaluate on validation set
    # ============================================================
    y_pred = model.predict(X_val_scaled)

    print("\n=== Validation Performance (NN) ===")
    metrics = compute_metrics(y_val, y_pred)
    print_metrics(metrics)

    # ============================================================
    # 6) Save model package (timestamped folder)
    # ============================================================
    save_model(
        model_name=MODEL_NAME,
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
