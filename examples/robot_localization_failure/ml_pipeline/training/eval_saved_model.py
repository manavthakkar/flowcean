#!/usr/bin/env python3
"""
Standalone eval script — loads a saved model and runs evaluation on eval.parquet.
Use this to test the OOM fix without retraining, and to generate plots/PDF report.

Usage (from robot_localization_failure dir):
    python -m ml_pipeline.training.eval_saved_model [--model MODEL_DIR_NAME]
    python -m ml_pipeline.training.eval_saved_model --report-only   # skip eval, just regenerate report
"""

import argparse
import gc
import json
from pathlib import Path

import joblib

import numpy as np
import polars as pl

from ml_pipeline.utils.paths import DATASETS, ARTIFACTS
from ml_pipeline.utils.common import (
    add_temporal_features,
    apply_scaler,
    compute_metrics,
    print_metrics,
    remove_leaky_columns,
)
from ml_pipeline.training.evaluation_plots import create_all_plots
from ml_pipeline.training.report_generator import generate_pdf_report

MODELS_DIR = ARTIFACTS / "models"


def load_model(model_dir: Path):
    model = joblib.load(model_dir / "model.pkl")
    scaler_path = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    with open(model_dir / "feature_columns.json") as f:
        feature_cols = json.load(f)
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, feature_cols, metadata


def run_eval(model_dir: Path):
    print(f"Loading model from: {model_dir}")
    model, scaler, feature_cols, metadata = load_model(model_dir)

    use_temporal = bool(metadata.get("temporal_features") or metadata.get("use_temporal_features"))
    print(f"  temporal_features: {use_temporal}")
    print(f"  feature_cols: {len(feature_cols)}")

    label_col = "lbl_win_02s"  # adjust if needed

    eval_path = DATASETS / "eval.parquet"
    print(f"\nLoading eval dataset: {eval_path}")
    df = pl.read_parquet(eval_path).drop_nulls()
    print(f"  Shape: {df.shape}")

    if use_temporal:
        print("Adding temporal features...")
        df = add_temporal_features(df)

    df = remove_leaky_columns(df)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in eval dataset: {missing}")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not in eval dataset. Available: {[c for c in df.columns if 'lbl' in c or 'delocal' in c]}")

    print(f"\nExtracting features ({len(feature_cols)} cols × {df.height} rows)...")
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y_true = df[label_col].to_numpy()
    del df
    gc.collect()
    print(f"  Memory freed (DataFrame deleted). X shape: {X.shape}, dtype: {X.dtype}")

    X_scaled = apply_scaler(X, scaler)
    del X
    gc.collect()
    print("  Scaled and freed raw X.")

    print("\nRunning predictions...")
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_scaled)
        y_proba = None

    print("\n=== EVAL METRICS (threshold=0.5) ===")
    metrics_t05 = compute_metrics(y_true, y_pred)
    print_metrics(metrics_t05)

    # Threshold sweep
    if y_proba is not None:
        print("\n=== THRESHOLD SWEEP ===")
        from sklearn.metrics import matthews_corrcoef
        best_mcc, best_thr, best_metrics = -1, 0.5, {}
        thresholds = np.arange(0.1, 0.95, 0.05)
        print(f"{'Thr':>6}  {'P':>6}  {'R':>6}  {'F1':>6}  {'MCC':>6}")
        for thr in thresholds:
            yp = (y_proba >= thr).astype(int)
            m = compute_metrics(y_true, yp)
            mcc = matthews_corrcoef(y_true, yp)
            print(f"  {thr:.2f}   {m['precision']:.3f}   {m['recall']:.3f}   {m['f1']:.3f}   {mcc:.3f}")
            if mcc > best_mcc:
                best_mcc, best_thr, best_metrics = mcc, thr, m
        print(f"\nBest threshold (MCC): {best_thr:.2f}  →  MCC={best_mcc:.4f}, "
              f"P={best_metrics['precision']:.4f}, R={best_metrics['recall']:.4f}, "
              f"F1={best_metrics['f1']:.4f}")

    # Save compact parquet
    out_path = model_dir / "eval_results.parquet"
    proba_col = y_proba.tolist() if y_proba is not None else [None] * len(y_pred)
    pl.DataFrame({
        label_col:     y_true.tolist(),
        "prediction":  y_pred.tolist(),
        "probability": proba_col,
    }).write_parquet(out_path)
    print(f"\n✔ Saved eval predictions → {out_path}")

    return model, feature_cols, metadata, y_true, y_proba, y_pred, metrics_t05, best_thr, best_metrics


def generate_report(model_dir: Path):
    """Generate plots + PDF report from a saved eval_results.parquet (no re-eval needed)."""
    model, scaler, feature_cols, metadata = load_model(model_dir)
    best_algo = metadata.get("selected_algorithm", "lgbm")

    results_path = model_dir / "eval_results.parquet"
    print(f"Loading saved eval results: {results_path}")
    df_res = pl.read_parquet(results_path)

    label_col = df_res.columns[0]  # first column is the label
    y_true  = df_res[label_col].to_numpy()
    y_pred  = df_res["prediction"].to_numpy()
    y_proba = df_res["probability"].to_numpy() if "probability" in df_res.columns else None

    from sklearn.metrics import matthews_corrcoef
    metrics_t05 = compute_metrics(y_true, y_pred)
    print_metrics(metrics_t05)

    best_mcc, best_thr, best_metrics = -1, 0.5, metrics_t05
    if y_proba is not None:
        for thr in np.arange(0.1, 0.95, 0.05):
            yp = (y_proba >= thr).astype(int)
            mcc = matthews_corrcoef(y_true, yp)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thr = float(thr)
                best_metrics = compute_metrics(y_true, yp)

    print(f"\nGenerating plots...")
    plot_paths = create_all_plots(
        y_true=y_true,
        y_proba=y_proba,
        y_pred_t05=y_pred,
        best_thr=best_thr,
        model=model,
        feature_cols=feature_cols,
        best_algo=best_algo,
        model_dir=model_dir,
    )
    svg_paths = {k: v for k, v in plot_paths.items() if str(v).endswith(".svg")}

    print(f"Generating PDF report...")
    generate_pdf_report(
        model_dir=model_dir,
        model_name=f"small_odometry_drift_{best_algo}",
        best_algo=best_algo,
        n_trials=metadata.get("n_trials", 0),
        metrics_t05=metrics_t05,
        best_thr=best_thr,
        best_metrics=best_metrics,
        train_maps=["unsym map 1"],
        eval_maps=["unsym map 1"],
        odometry="[0.25, 0.25, 0.1, 0.25, 0.25, 0.1]",
        notes="Small odom drift in theta",
        feature_flags={
            "temporal": bool(metadata.get("temporal_features")),
            "scanmap":  bool(metadata.get("use_scanmap_features", True)),
            "particle": bool(metadata.get("use_particle_features", True)),
            "amcl":     bool(metadata.get("use_amcl_pose", False)),
        },
        svg_paths=svg_paths,
    )
    print(f"✔ Report saved → {model_dir / 'report' / 'model_report.pdf'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="small_odometry_drift_lgbm_2026-03-04_16-24-27",
                        help="Model directory name inside artifacts/models/")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip eval, load saved eval_results.parquet and regenerate plots/report only")
    args = parser.parse_args()

    model_dir = MODELS_DIR / args.model
    if not model_dir.exists():
        available = [d.name for d in MODELS_DIR.iterdir() if d.is_dir()]
        raise ValueError(f"Model dir not found: {model_dir}\nAvailable: {available}")

    if args.report_only:
        generate_report(model_dir)
    else:
        run_eval(model_dir)
        generate_report(model_dir)


if __name__ == "__main__":
    main()
