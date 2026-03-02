# ml_pipeline/evaluation/evaluate_model.py

import argparse
import json
import joblib
import polars as pl
import numpy as np
from pathlib import Path

from sklearn.metrics import fbeta_score
import matplotlib.pyplot as plt

from ml_pipeline.utils.paths import DATASETS, MODELS
from ml_pipeline.utils.common import apply_scaler, add_temporal_features


def compute_f1_vs_threshold(y_true, y_proba):
    """Compute F1 scores for various decision thresholds."""
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = []

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f1 = fbeta_score(y_true, y_pred, beta=1.0, zero_division=0)
        f1_scores.append(f1)

    return thresholds, f1_scores


def plot_f1_vs_threshold_combined(curves_data, output_path: Path):
    """Plot multiple F1 vs threshold curves on the same plot and save as SVG.

    Args:
        curves_data: List of tuples (thresholds, f1_scores, legend_text)
        output_path: Path to save the SVG file
    """
    plt.figure(figsize=(8, 5))

    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]

    for i, (thresholds, f1_scores, legend_text) in enumerate(curves_data):
        color = colors[i % len(colors)]
        plt.plot(thresholds, f1_scores, linewidth=2, color=color, label=legend_text)

    plt.xlabel("Decision Threshold", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.title("F1 Score vs Decision Threshold", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save as SVG
    plt.savefig(output_path, format="svg")
    plt.close()

    print(f"\n📊 Combined F1 vs Threshold plot saved → {output_path}")


def load_model_package(model_dir: Path):
    """Load model, scaler, feature list, and optional metadata from a given directory."""
    model = joblib.load(model_dir / "model.pkl")

    scaler_path = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    with open(model_dir / "feature_columns.json", "r") as f:
        feature_cols = json.load(f)

    metadata_path = model_dir / "metadata.json"
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    return model, scaler, feature_cols, metadata


def select_model_with_legend(model_dirs, model_number: int):
    """Select a model and get legend text from user input."""
    print(f"\n--- Model {model_number} ---")
    print("Available models:")
    for i, d in enumerate(model_dirs):
        print(f"[{i}] {d.name}")

    idx = input(f"Select model {model_number} index: ").strip()
    if not idx.isdigit() or int(idx) not in range(len(model_dirs)):
        raise ValueError("Invalid selection.")

    model_dir = model_dirs[int(idx)]
    legend_text = input(f"Enter legend text for model {model_number}: ").strip()

    if not legend_text:
        legend_text = model_dir.name

    return model_dir, legend_text


def get_predictions_for_model(model_dir: Path, df: pl.DataFrame):
    """Load model and get predictions for the evaluation dataset."""
    print(f"\n📦 Processing model: {model_dir.name}")

    model, scaler, feature_cols, metadata = load_model_package(model_dir)

    # Check if model needs temporal features
    use_temporal = False
    if metadata is not None:
        use_temporal = bool(
            metadata.get("use_temporal_features") or metadata.get("temporal_features")
        )

    df_model = df.clone()
    if use_temporal:
        print("🔧 Model expects TEMPORAL features → adding them...")
        df_model = add_temporal_features(df_model)
    else:
        print("ℹ️ Model does NOT use temporal features.")

    # Check required columns
    missing = [c for c in feature_cols if c not in df_model.columns]
    if missing:
        raise ValueError(f"❌ Missing columns in eval dataset: {missing}")

    X = df_model.select(feature_cols).to_numpy()
    y_true = df_model["is_delocalized"].to_numpy()

    # Scale if needed
    X_scaled = apply_scaler(X, scaler)

    # Get predictions
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)[:, 1]
    else:
        print("⚠️ Model has no predict_proba.")
        y_proba = None

    return y_true, y_proba


def main():
    parser = argparse.ArgumentParser(
        description="Compare F1 vs Threshold curves for two models."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="f1_vs_threshold_comparison.svg",
        help="Output filename for the combined plot (default: f1_vs_threshold_comparison.svg)",
    )
    args = parser.parse_args()

    # ---------------------------
    # Get available models
    # ---------------------------
    model_dirs = sorted([d for d in MODELS.iterdir() if d.is_dir()])

    if not model_dirs:
        raise RuntimeError("❌ No model directories found in artifacts/models/")

    # ---------------------------
    # Select two models with legend text
    # ---------------------------
    model_dir_1, legend_1 = select_model_with_legend(model_dirs, 1)
    model_dir_2, legend_2 = select_model_with_legend(model_dirs, 2)

    # ---------------------------
    # Load evaluation dataset
    # ---------------------------
    eval_path = DATASETS / "eval.parquet"
    print(f"\nReading evaluation dataset: {eval_path}")
    df = pl.read_parquet(eval_path).drop_nulls()

    # ---------------------------
    # Get predictions for both models
    # ---------------------------
    y_true_1, y_proba_1 = get_predictions_for_model(model_dir_1, df)
    y_true_2, y_proba_2 = get_predictions_for_model(model_dir_2, df)

    # ---------------------------
    # Compute F1 curves
    # ---------------------------
    curves_data = []

    if y_proba_1 is not None:
        thresholds_1, f1_scores_1 = compute_f1_vs_threshold(y_true_1, y_proba_1)
        curves_data.append((thresholds_1, f1_scores_1, legend_1))
    else:
        print("⚠️ Skipping model 1: no predict_proba support.")

    if y_proba_2 is not None:
        thresholds_2, f1_scores_2 = compute_f1_vs_threshold(y_true_2, y_proba_2)
        curves_data.append((thresholds_2, f1_scores_2, legend_2))
    else:
        print("⚠️ Skipping model 2: no predict_proba support.")

    if not curves_data:
        print("❌ No models with predict_proba available. Cannot plot.")
        return

    # ---------------------------
    # Plot combined curves
    # ---------------------------
    output_path = MODELS / args.output
    plot_f1_vs_threshold_combined(curves_data, output_path)


if __name__ == "__main__":
    main()
