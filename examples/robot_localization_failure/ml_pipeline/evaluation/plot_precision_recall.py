# ml_pipeline/evaluation/plot_precision_recall.py

import json
import joblib
import polars as pl
import numpy as np
from pathlib import Path

from sklearn.metrics import precision_recall_curve, average_precision_score

from ml_pipeline.utils.paths import DATASETS, MODELS
from ml_pipeline.utils.common import apply_scaler, add_temporal_features


def load_model_package(model_dir: Path):
    model = joblib.load(model_dir / "model.pkl")

    scaler_path = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    with open(model_dir / "feature_columns.json", "r") as f:
        feature_cols = json.load(f)

    metadata = None
    meta_path = model_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    return model, scaler, feature_cols, metadata


def select_model(model_dirs, prompt):
    print("\nAvailable models:")
    for i, d in enumerate(model_dirs):
        print(f"[{i}] {d.name}")
    idx = input(prompt).strip()
    if not idx.isdigit() or int(idx) not in range(len(model_dirs)):
        raise ValueError("Invalid selection.")
    return model_dirs[int(idx)]


def compute_pr_curve(model_dir: Path, df: pl.DataFrame):
    model, scaler, feature_cols, metadata = load_model_package(model_dir)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            f"Model {model_dir.name} has no predict_proba; cannot plot PR curve."
        )

    use_temporal = (
        metadata is not None
        and (
            metadata.get("temporal_features", False)
            or metadata.get("use_temporal_features", False)
        )
    )

    df_in = df
    if use_temporal:
        print(f"🔧 {model_dir.name}: adding temporal features...")
        df_in = add_temporal_features(df_in)
    else:
        print(f"ℹ️ {model_dir.name}: no temporal features.")

    missing = [c for c in feature_cols if c not in df_in.columns]
    if missing:
        raise ValueError(f"❌ {model_dir.name} missing columns: {missing}")

    X = df_in.select(feature_cols).to_numpy()
    y_true = df_in["is_delocalized"].to_numpy()

    X_scaled = apply_scaler(X, scaler)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    return precision_vals, recall_vals, avg_precision


def main():
    # ---------------------------
    # choose models
    # ---------------------------
    model_dirs = [d for d in MODELS.iterdir() if d.is_dir()]
    if not model_dirs:
        raise RuntimeError("No model directories under artifacts/models")

    model_dir_a = select_model(model_dirs, "Select first model index: ")
    label_a = input("Label for first model (leave blank to use model name): ").strip()
    model_dir_b = select_model(model_dirs, "Select second model index: ")
    label_b = input("Label for second model (leave blank to use model name): ").strip()
    show_baseline = (
        input("Show baseline line? [y/N]: ").strip().lower() in {"y", "yes"}
    )

    if not label_a:
        label_a = model_dir_a.name
    if not label_b:
        label_b = model_dir_b.name

    print(f"\n📦 Using model A: {model_dir_a.name}")
    print(f"🏷️  Label A: {label_a}")
    print(f"📦 Using model B: {model_dir_b.name}")
    print(f"🏷️  Label B: {label_b}")

    # ---------------------------
    # load eval data
    # ---------------------------
    eval_path = DATASETS / "eval.parquet"
    print(f"Reading evaluation dataset: {eval_path}")
    df = pl.read_parquet(eval_path).drop_nulls()

    pr_a = compute_pr_curve(model_dir_a, df)
    pr_b = compute_pr_curve(model_dir_b, df)

    # ---------------------------
    # plot precision-recall curves
    # ---------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save the PR curve plot.") from exc

    (precision_a, recall_a, ap_a) = pr_a
    (precision_b, recall_b, ap_b) = pr_b
    out_path = model_dir_a.parent / "precision_recall_curve_compare.svg"

    plt.figure(figsize=(6, 4))
    plt.plot(
        recall_a,
        precision_a,
        label=f"{label_a} (AP={ap_a:.3f})",
        color="#1f77b4",
    )
    plt.plot(
        recall_b,
        precision_b,
        label=f"{label_b} (AP={ap_b:.3f})",
        color="#ff7f0e",
    )
    if show_baseline:
        baseline = float(np.mean(df["is_delocalized"].to_numpy()))
        plt.hlines(
            baseline,
            xmin=0.0,
            xmax=1.0,
            colors="#999999",
            linestyles="--",
            label=f"Baseline (pos rate={baseline:.3f})",
        )
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"\n Saved comparison PR curve → {out_path}")


if __name__ == "__main__":
    main()
