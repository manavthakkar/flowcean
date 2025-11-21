# ml_pipeline/evaluation/evaluate_model.py

import argparse
import json
import joblib
import polars as pl
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    fbeta_score,
)

from ml_pipeline.utils.paths import DATASETS, MODELS
from ml_pipeline.utils.common import apply_scaler


# ============================================================
# Helper: load model package
# ============================================================
def load_model_package(model_dir: Path):
    """Load model, scaler and feature list from a given directory."""
    model = joblib.load(model_dir / "model.pkl")

    scaler_path = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    with open(model_dir / "feature_columns.json", "r") as f:
        feature_cols = json.load(f)

    return model, scaler, feature_cols


def threshold_search(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Grid-search over thresholds to maximize F0.5.

    Returns a dict with:
      - best_thr
      - best_precision
      - best_recall
      - best_f1
      - best_f05
    """
    # Default: evaluate threshold=0.5 as baseline
    base_pred = (y_proba >= 0.5).astype(int)
    base_prec = precision_score(y_true, base_pred)
    base_rec = recall_score(y_true, base_pred)
    base_f1 = fbeta_score(y_true, base_pred, beta=1.0)
    base_f05 = fbeta_score(y_true, base_pred, beta=0.5)

    best = {
        "best_thr": 0.5,
        "best_precision": base_prec,
        "best_recall": base_rec,
        "best_f1": base_f1,
        "best_f05": base_f05,
    }

    # Search a grid of thresholds
    thresholds = np.linspace(0.05, 0.95, 19)

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = fbeta_score(y_true, y_pred, beta=1.0)
        f05 = fbeta_score(y_true, y_pred, beta=0.5)

        if f05 > best["best_f05"]:
            best = {
                "best_thr": float(thr),
                "best_precision": prec,
                "best_recall": rec,
                "best_f1": f1,
                "best_f05": f05,
            }

    return best


# ============================================================
# Main evaluation entry point
# ============================================================
def main():
    # ---------------------------
    # Parse arguments
    # ---------------------------
    parser = argparse.ArgumentParser(description="Evaluate trained ML model.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Name of the model directory inside artifacts/models/",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eval",
        help="Name of dataset parquet in artifacts/datasets/ "
             "(e.g. 'eval' or 'eval_temporal').",
    )
    args = parser.parse_args()

    # ---------------------------
    # Resolve model directory
    # ---------------------------
    model_dirs = [d for d in MODELS.iterdir() if d.is_dir()]

    if not model_dirs:
        raise RuntimeError("❌ No model directories found in artifacts/models/")

    # If user provided a model directory name explicitly
    if args.model_dir is not None:
        candidate = MODELS / args.model_dir
        if not candidate.exists():
            print("❌ Requested model does not exist.")
            print("Available models:")
            for d in model_dirs:
                print(" -", d.name)
            return
        model_dir = candidate
    else:
        # No --model_dir → ask which model to use
        print("\nAvailable models:")
        for i, d in enumerate(model_dirs):
            print(f"[{i}] {d.name}")

        idx = input("Select a model index: ").strip()
        if not idx.isdigit() or int(idx) not in range(len(model_dirs)):
            raise ValueError("Invalid selection.")
        model_dir = model_dirs[int(idx)]

    print(f"\n📦 Using model: {model_dir.name}")

    # Load model/scaler/features
    model, scaler, feature_cols = load_model_package(model_dir)

    # ---------------------------
    # Load evaluation dataset
    # ---------------------------
    eval_path = DATASETS / f"{args.dataset}.parquet"
    print(f"Reading evaluation dataset: {eval_path}")

    if not eval_path.exists():
        raise FileNotFoundError(
            f"❌ Evaluation dataset not found: {eval_path}\n"
            f"Make sure you created it (e.g. eval.parquet, eval_temporal.parquet)."
        )

    df = pl.read_parquet(eval_path).drop_nulls()

    # Check required columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing columns in eval dataset: {missing}")

    X = df.select(feature_cols).to_numpy()
    y_true = df["is_delocalized"].to_numpy()

    # Scale features (if scaler is not None)
    X_scaled = apply_scaler(X, scaler)

    # Predict probabilities if available
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)[:, 1]
    else:
        y_proba = None

    # ===========================
    # Metrics at default threshold 0.5
    # ===========================
    if y_proba is not None:
        y_pred_05 = (y_proba >= 0.5).astype(int)
    else:
        # Fallback to model.predict (no probability interface)
        y_pred_05 = model.predict(X_scaled)

    print("\n=== Evaluation Metrics @ threshold = 0.5 ===")
    print(classification_report(y_true, y_pred_05))

    precision_05 = precision_score(y_true, y_pred_05)
    recall_05 = recall_score(y_true, y_pred_05)
    f1_05 = fbeta_score(y_true, y_pred_05, beta=1.0)
    f05_05 = fbeta_score(y_true, y_pred_05, beta=0.5)
    cm_05 = confusion_matrix(y_true, y_pred_05)

    print(f"Precision:  {precision_05:.4f}")
    print(f"Recall:     {recall_05:.4f}")
    print(f"F1 Score:   {f1_05:.4f}")
    print(f"F0.5 Score: {f05_05:.4f}")

    print("\n=== Confusion Matrix @ threshold = 0.5 ===")
    print(cm_05)

    # ===========================
    # Threshold search for best F0.5
    # ===========================
    if y_proba is not None:
        best = threshold_search(y_true, y_proba)

        print("\n=== Threshold search (optimize F0.5) ===")
        print(f"Best threshold: {best['best_thr']:.3f}")
        print(f"Precision:      {best['best_precision']:.4f}")
        print(f"Recall:         {best['best_recall']:.4f}")
        print(f"F1 Score:       {best['best_f1']:.4f}")
        print(f"F0.5 Score:     {best['best_f05']:.4f}")

        # Recompute confusion matrix at best threshold
        y_pred_best = (y_proba >= best["best_thr"]).astype(int)
        cm_best = confusion_matrix(y_true, y_pred_best)

        print(f"\n=== Confusion Matrix @ best threshold ({best['best_thr']:.3f}) ===")
        print(cm_best)

        # This will be the final prediction we save
        final_pred = y_pred_best
        final_thr = best["best_thr"]
    else:
        print("\n⚠ Model has no predict_proba; cannot optimize threshold. "
              "Using decision function / default predictions.")
        final_pred = y_pred_05
        final_thr = 0.5

    # ---------------------------
    # Save predictions
    # ---------------------------
    out_path = model_dir / f"eval_results_{args.dataset}.parquet"
    df_out = df.with_columns([
        pl.Series("prediction", final_pred),
        pl.Series("probability", y_proba if y_proba is not None else [None] * len(final_pred)),
        pl.lit(final_thr).alias("used_threshold"),
    ])
    df_out.write_parquet(out_path)

    print(f"\n✔ Saved predictions → {out_path}")


if __name__ == "__main__":
    main()
