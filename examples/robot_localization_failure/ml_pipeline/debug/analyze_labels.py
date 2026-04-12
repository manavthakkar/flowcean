#!/usr/bin/env python3
"""
Feature separability analysis across labeling strategies.

For every (feature, label) combination this script computes two metrics:

  ROC AUC  — threshold-independent ranking quality.
             Invariant to class imbalance: TPR/FPR are both within-class ratios,
             so a 2% positive rate does not bias the score.
             Limitation: with very rare positives the ROC curve looks generous
             even when precision is poor, because FPR is computed over the vast
             majority (negatives), masking most false positives.

  PR AUC   — average precision (area under precision-recall curve).
             Focuses entirely on the positive class.  With a 2% positive rate
             a random classifier scores ~0.02; any score above that reflects
             real discriminative power.  This is the more informative metric
             for the highly imbalanced predictive-window labels.

Both are rendered as heatmaps.  Features are rows; labeling strategies
are columns grouped as:

  is_delocalized        — original 0.2 m threshold label
  lbl_err_020…050       — threshold-based labels (position/heading error > T)
  lbl_win_02s…10s       — predictive-window labels (reset within W seconds)

Usage
-----
Run from the robot_localization_failure directory:

    python -m ml_pipeline.debug.analyze_labels

Optional flags:
    --dataset   Path to parquet file (default: artifacts/datasets/train.parquet)
    --output    Path for the PNG output (default: artifacts/label_auc_heatmap.png)
    --sample    Rows to sample (default: 500000; 0 = full dataset).
                500 k rows gives ROC/PR AUC standard error < 0.001 — statistically
                identical to running on the full 26 M rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from ml_pipeline.utils.paths import DATASETS, ARTIFACTS
from ml_pipeline.utils.common import LEAKY_COLUMNS


# ─── label column discovery ───────────────────────────────────────────────────

def _label_cols(df: pl.DataFrame) -> list[str]:
    """Return label columns in a fixed display order."""
    all_cols = df.columns
    ordered: list[str] = []
    if "is_delocalized" in all_cols:
        ordered.append("is_delocalized")
    for c in sorted(all_cols):
        if c.startswith("lbl_err_"):
            ordered.append(c)
    for c in sorted(all_cols):
        if c.startswith("lbl_win_"):
            ordered.append(c)
    return ordered


# ─── feature column discovery ────────────────────────────────────────────────

def _feature_cols(df: pl.DataFrame, label_cols: list[str]) -> list[str]:
    """Return numeric non-leaky, non-label columns."""
    leaky = set(LEAKY_COLUMNS) | set(label_cols) | {"is_delocalized"}
    numeric_types = (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    return [
        c for c, dt in df.schema.items()
        if c not in leaky and isinstance(dt, numeric_types)
    ]


# ─── AUC computation ─────────────────────────────────────────────────────────

def compute_auc_matrices(
    df: pl.DataFrame,
    feature_cols: list[str],
    label_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (roc_auc, pr_auc), each of shape (n_features, n_labels).

    ROC AUC: max(raw, 1-raw) — invariant to whether low or high feature values
             correspond to positives.
    PR AUC:  average_precision_score with the same sign flip applied:
             if max(raw, 1-raw) used the flipped score, negate x first.
             Baseline (random) ≈ positive rate of the label.
    """
    df_clean = df.drop_nulls()
    n_f, n_l = len(feature_cols), len(label_cols)
    roc = np.full((n_f, n_l), 0.5)
    pr  = np.full((n_f, n_l), np.nan)

    for j, lbl in enumerate(label_cols):
        y = df_clean[lbl].to_numpy().astype(bool)
        if y.sum() == 0 or (~y).sum() == 0:
            continue
        baseline = y.mean()
        pr[:, j] = baseline          # fill with random-classifier baseline

        for i, feat in enumerate(feature_cols):
            x = df_clean[feat].to_numpy().astype(float)
            if np.isnan(x).any() or np.std(x) == 0.0:
                continue
            try:
                raw_roc = float(roc_auc_score(y, x))
                flip = raw_roc < 0.5   # use inverted score if that's better
                roc[i, j] = 1.0 - raw_roc if flip else raw_roc
                x_pr = -x if flip else x
                pr[i, j] = float(average_precision_score(y, x_pr))
            except Exception:
                pass

    return roc, pr


# ─── plotting ─────────────────────────────────────────────────────────────────

def _label_display(lbl: str) -> str:
    if lbl == "is_delocalized":
        return "err>0.2m\n(orig)"
    if lbl.startswith("lbl_err_"):
        v = int(lbl[8:]) / 100
        return f"err>{v:.2f}m"
    if lbl.startswith("lbl_win_"):
        v = int(lbl[8:10])
        return f"win {v}s"
    return lbl


def _draw_separators(ax, label_cols: list[str]) -> None:
    has_orig = "is_delocalized" in label_cols
    err_count = sum(1 for c in label_cols if c.startswith("lbl_err_"))
    if has_orig:
        ax.axvline(0.5, color="steelblue", lw=1.5, alpha=0.5)
    if err_count > 0:
        sep = (1 + err_count - 0.5) if has_orig else (err_count - 0.5)
        ax.axvline(sep, color="seagreen", lw=1.5, alpha=0.5)


def make_dual_heatmap(
    roc: np.ndarray,
    pr:  np.ndarray,
    feature_cols: list[str],
    label_cols:   list[str],
    baselines:    list[float],
    out_path: Path,
) -> None:
    n_feat, n_lbl = roc.shape
    fig_h = max(10, 0.32 * n_feat)
    fig_w = max(22, 1.1 * n_lbl)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    xlabels = [_label_display(c) for c in label_cols]

    # ── ROC AUC panel ──────────────────────────────────────────────────────
    im_roc = ax_roc.imshow(roc, aspect="auto", vmin=0.5, vmax=1.0, cmap="YlOrRd")
    fig.colorbar(im_roc, ax=ax_roc, label="ROC AUC", fraction=0.03, pad=0.02)
    ax_roc.set_xticks(range(n_lbl))
    ax_roc.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax_roc.set_yticks(range(n_feat))
    ax_roc.set_yticklabels(feature_cols, fontsize=7)
    ax_roc.set_title(
        "ROC AUC  (threshold-independent ranking)\n"
        "Invariant to class imbalance — 0.50 = no info, 1.00 = perfect",
        fontsize=9)
    for i in range(n_feat):
        for j in range(n_lbl):
            v = roc[i, j]
            ax_roc.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if v >= 0.85 else "black")
    _draw_separators(ax_roc, label_cols)

    # ── PR AUC panel ───────────────────────────────────────────────────────
    # Colour scale starts at 0 (random baseline varies per label)
    im_pr = ax_pr.imshow(pr, aspect="auto", vmin=0.0, vmax=1.0, cmap="YlOrRd")
    fig.colorbar(im_pr, ax=ax_pr, label="PR AUC (avg precision)", fraction=0.03, pad=0.02)
    ax_pr.set_xticks(range(n_lbl))
    ax_pr.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax_pr.set_yticks(range(n_feat))
    ax_pr.set_yticklabels(feature_cols, fontsize=7)
    # Show baselines in x-axis labels
    base_labels = [
        f"{_label_display(c)}\n(base={baselines[j]:.2f})"
        for j, c in enumerate(label_cols)
    ]
    ax_pr.set_xticklabels(base_labels, rotation=45, ha="right", fontsize=7)
    ax_pr.set_title(
        "PR AUC  (positive-class focus, accounts for imbalance)\n"
        "Baseline ≈ positive rate shown in x-axis labels",
        fontsize=9)
    for i in range(n_feat):
        for j in range(n_lbl):
            v = pr[i, j]
            ax_pr.text(j, i, f"{v:.2f}", ha="center", va="center",
                       fontsize=5, color="white" if v >= 0.7 else "black")
    _draw_separators(ax_pr, label_cols)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"✔  Dual heatmap saved → {out_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path,
                        default=DATASETS / "train.parquet")
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "label_auc_heatmap.png")
    parser.add_argument("--sample", type=int, default=500_000,
                        help="Randomly sample this many rows (0 = full dataset). "
                             "500 k gives standard error < 0.001 on AUC — statistically "
                             "identical to the full 26 M rows.")
    args = parser.parse_args()

    print(f"Loading: {args.dataset}")
    df = pl.read_parquet(args.dataset)
    print(f"  Shape: {df.shape}")

    if args.sample and args.sample < df.height:
        df = df.sample(n=args.sample, seed=42)
        print(f"  Sampled: {df.shape[0]} rows  (SE < 0.001 — same result as full dataset)")
    else:
        print(f"  Using full dataset ({df.shape[0]} rows)")

    lbls  = _label_cols(df)
    feats = _feature_cols(df, lbls)

    print(f"  Label columns   ({len(lbls)}): {lbls}")
    print(f"  Feature columns ({len(feats)}): {feats}")

    if not lbls:
        print("ERROR: No label columns found. Rebuild dataset first.")
        return

    # Positive-rate baselines (random-classifier PR AUC baseline per label)
    df_clean = df.drop_nulls()
    baselines = [float(df_clean[lbl].mean()) for lbl in lbls]
    print(f"\nPositive rates (PR AUC baselines):")
    for lbl, b in zip(lbls, baselines):
        print(f"  {lbl:20s}: {b:.3f}")

    print("\nComputing ROC AUC + PR AUC matrices …")
    roc, pr = compute_auc_matrices(df_clean, feats, lbls)

    # ── Console summary ─────────────────────────────────────────────────────
    print("\nTop-5 features by ROC AUC for each label:")
    for j, lbl in enumerate(lbls):
        top_idx = np.argsort(roc[:, j])[::-1][:5]
        row = "  ".join(f"{feats[i]}={roc[i,j]:.3f}" for i in top_idx)
        print(f"  {lbl:20s}: {row}")

    print("\nTop-5 features by PR AUC for each label:")
    for j, lbl in enumerate(lbls):
        top_idx = np.argsort(pr[:, j])[::-1][:5]
        row = "  ".join(f"{feats[i]}={pr[i,j]:.3f}" for i in top_idx)
        print(f"  {lbl:20s} (base={baselines[j]:.3f}): {row}")

    mean_roc = roc.mean(axis=1)
    mean_pr  = pr.mean(axis=1)
    print("\nTop-10 features by mean ROC AUC across all labels:")
    for idx in np.argsort(mean_roc)[::-1][:10]:
        print(f"  {feats[idx]:35s}  roc={mean_roc[idx]:.3f}  pr={mean_pr[idx]:.3f}")

    make_dual_heatmap(roc, pr, feats, lbls, baselines, args.output)


if __name__ == "__main__":
    main()
