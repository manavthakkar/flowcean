# ml_pipeline/evaluation/plot_confusion_matrix.py

import json
import joblib
import polars as pl
import numpy as np
from pathlib import Path

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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


def main():
    # ---------------------------
    # choose model
    # ---------------------------
    model_dirs = [d for d in MODELS.iterdir() if d.is_dir()]
    if not model_dirs:
        raise RuntimeError("No model directories under artifacts/models")

    model_dir = select_model(model_dirs, "Select model index: ")
    print(f"\n📦 Using model: {model_dir.name}")

    model, scaler, feature_cols, metadata = load_model_package(model_dir)

    # ---------------------------
    # load eval data
    # ---------------------------
    eval_path = DATASETS / "eval.parquet"
    print(f"Reading evaluation dataset: {eval_path}")
    df = pl.read_parquet(eval_path).drop_nulls()

    use_temporal = (
        metadata is not None
        and (
            metadata.get("temporal_features", False)
            or metadata.get("use_temporal_features", False)
        )
    )
    if use_temporal:
        print("🔧 Model expects TEMPORAL features → adding them to eval dataset...")
        df = add_temporal_features(df)
    else:
        print("ℹ️ Model does NOT use temporal features.")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing columns in eval dataset: {missing}")

    X = df.select(feature_cols).to_numpy()
    y_true = df["is_delocalized"].to_numpy()

    X_scaled = apply_scaler(X, scaler)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        print("⚠️ Model has no predict_proba → using predict().")
        y_pred = model.predict(X_scaled)

    # ---------------------------
    # normalized confusion matrix
    # ---------------------------
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save the confusion matrix plot.") from exc

    labels = ["False", "True"]
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=labels,
    )

    disp.plot(
        cmap="Blues",
        values_format=".2f",
        ax=ax,
        colorbar=True,
    )

    # Match styling from save_confusion_matrix.py
    ax.set_title("", fontsize=18)
    ax.set_xlabel("Predicted label", fontsize=18)
    ax.set_ylabel("True label", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)

    for text in disp.text_.ravel():
        text.set_fontsize(18)

    fig.tight_layout()

    save_name = input("Save name (without extension): ").strip()
    if not save_name:
        save_name = "confusion_matrix_normalized"

    out_path = model_dir / f"{save_name}.svg"
    fig.savefig(out_path, dpi=300, format="svg")
    plt.close(fig)

    print(f"\n🧩 Saved normalized confusion matrix → {out_path}")


if __name__ == "__main__":
    main()
