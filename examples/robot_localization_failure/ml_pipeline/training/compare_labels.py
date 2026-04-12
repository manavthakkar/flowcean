#!/usr/bin/env python3
"""
Compare model performance across different labeling strategies.

Trains a CatBoost classifier (fixed hyperparameters, no Optuna) for each
label column in the dataset and reports a comparison table with:
  precision / recall / F1 / MCC / positive rate

Label columns evaluated:
  is_delocalized        — original 0.8 m threshold
  lbl_err_020…050       — threshold-based labels
  lbl_win_02s…10s       — predictive-window labels

The comparison table is saved to artifacts/label_comparison.csv and
printed to stdout.

Usage
-----
Run from the robot_localization_failure directory:

    python -m ml_pipeline.training.compare_labels

Optional flags:
    --train     Path to train parquet  (default: artifacts/datasets/train.parquet)
    --eval      Path to eval parquet   (default: artifacts/datasets/eval.parquet)
    --output    Path for CSV output    (default: artifacts/label_comparison.csv)
    --iterations  CatBoost iterations  (default: 400)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from ml_pipeline.utils.paths import DATASETS, ARTIFACTS
from ml_pipeline.utils.common import prepare_features


# ─── helpers ──────────────────────────────────────────────────────────────────

def _label_cols(df: pl.DataFrame) -> list[str]:
    """Return label columns in display order."""
    cols = df.columns
    ordered: list[str] = []
    if "is_delocalized" in cols:
        ordered.append("is_delocalized")
    for c in sorted(cols):
        if c.startswith("lbl_err_"):
            ordered.append(c)
    for c in sorted(cols):
        if c.startswith("lbl_win_"):
            ordered.append(c)
    return ordered


def _prepare(df: pl.DataFrame, label_col: str):
    """Return X, y numpy arrays for the given label column."""
    X, y, _ = prepare_features(df, label_col=label_col)
    return X.astype(np.float32), y.astype(bool)


def _train_and_eval(
    X_tr, y_tr,
    X_ev, y_ev,
    iterations: int,
    label_col: str,
) -> dict:
    pos_rate_train = float(y_tr.mean())
    pos_rate_eval  = float(y_ev.mean())

    if y_tr.sum() == 0 or (~y_tr).sum() == 0:
        return {
            "label": label_col,
            "pos_rate_train": pos_rate_train,
            "pos_rate_eval":  pos_rate_eval,
            "precision": float("nan"),
            "recall":    float("nan"),
            "f1":        float("nan"),
            "mcc":       float("nan"),
            "note": "degenerate — all one class",
        }

    # Class weights to handle imbalance
    neg, pos = (~y_tr).sum(), y_tr.sum()
    scale_pos = neg / pos if pos > 0 else 1.0

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        scale_pos_weight=scale_pos,
        eval_metric="F1",
        random_seed=42,
        verbose=0,
    )
    model.fit(X_tr, y_tr.astype(int))
    y_pred = model.predict(X_ev).astype(bool)

    return {
        "label":          label_col,
        "pos_rate_train": pos_rate_train,
        "pos_rate_eval":  pos_rate_eval,
        "precision":      float(precision_score(y_ev, y_pred, zero_division=0)),
        "recall":         float(recall_score(y_ev, y_pred, zero_division=0)),
        "f1":             float(f1_score(y_ev, y_pred, zero_division=0)),
        "mcc":            float(matthews_corrcoef(y_ev, y_pred)),
        "note": "",
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train",      type=Path, default=DATASETS / "train.parquet")
    parser.add_argument("--eval",       type=Path, default=DATASETS / "eval.parquet")
    parser.add_argument("--output",     type=Path, default=ARTIFACTS / "label_comparison.csv")
    parser.add_argument("--iterations", type=int,  default=400)
    parser.add_argument("--sample", type=int, default=0,
                        help="Randomly sample this many rows from the training set "
                             "(0 = use full dataset)")
    args = parser.parse_args()

    print(f"Loading train: {args.train}")
    train_df = pl.read_parquet(args.train).drop_nulls()
    if args.sample and args.sample < train_df.height:
        train_df = train_df.sample(n=args.sample, seed=42)
        print(f"  Sampled train: {train_df.shape[0]} rows")
    print(f"Loading eval : {args.eval}")
    eval_df  = pl.read_parquet(args.eval).drop_nulls()

    lbls = _label_cols(train_df)
    # Keep only labels that also exist in the eval set
    lbls = [l for l in lbls if l in eval_df.columns]

    if not lbls:
        print("ERROR: No shared label columns found. Rebuild the dataset first.")
        return

    print(f"\n{'Label column':<22} {'pos_tr':>6} {'pos_ev':>6}  "
          f"{'prec':>6} {'rec':>6} {'F1':>6} {'MCC':>6}")
    print("-" * 75)

    results: list[dict] = []
    for lbl in lbls:
        try:
            X_tr, y_tr = _prepare(train_df, lbl)
            X_ev, y_ev = _prepare(eval_df,  lbl)
        except ValueError as e:
            print(f"  {lbl:<22} — skipped ({e})")
            continue

        row = _train_and_eval(X_tr, y_tr, X_ev, y_ev, args.iterations, lbl)
        results.append(row)

        note = f"  [{row['note']}]" if row["note"] else ""
        print(
            f"  {lbl:<22}"
            f"  {row['pos_rate_train']:>5.1%}"
            f"  {row['pos_rate_eval']:>5.1%}"
            f"  {row['precision']:>6.3f}"
            f"  {row['recall']:>6.3f}"
            f"  {row['f1']:>6.3f}"
            f"  {row['mcc']:>6.3f}"
            f"{note}"
        )

    # Save CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "pos_rate_train", "pos_rate_eval",
                  "precision", "recall", "f1", "mcc", "note"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✔  Results saved → {args.output}")

    # Highlight best by MCC
    valid = [r for r in results if not np.isnan(r["mcc"])]
    if valid:
        best = max(valid, key=lambda r: r["mcc"])
        print(f"\nBest MCC: {best['label']}  (MCC={best['mcc']:.3f})")

    # Highlight best by F1
    if valid:
        best_f1 = max(valid, key=lambda r: r["f1"])
        print(f"Best F1 : {best_f1['label']}  (F1={best_f1['f1']:.3f})")


if __name__ == "__main__":
    main()
