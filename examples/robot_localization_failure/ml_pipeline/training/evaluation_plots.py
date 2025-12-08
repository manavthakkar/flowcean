# ml_pipeline/training/evaluation_plots.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    f1_score,
)

# ============================================================
# UTILITIES
# ============================================================

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_plot(fig, out_dir: Path, filename: str) -> Dict[str, Path]:
    """
    Save both PNG + SVG versions of a plot, return paths.
    """
    _ensure_dir(out_dir)
    png_path = out_dir / f"{filename}.png"
    svg_path = out_dir / f"{filename}.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")

    plt.close(fig)

    return {"png": png_path, "svg": svg_path}


# ============================================================
# ROC CURVE
# ============================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_dir: Path,
) -> Optional[Dict[str, Path]]:
    if y_proba is None:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    return _save_plot(fig, out_dir, "roc_curve")


# ============================================================
# PRECISION–RECALL CURVE
# ============================================================

def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_dir: Path,
) -> Optional[Dict[str, Path]]:
    if y_proba is None:
        return None

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend()

    return _save_plot(fig, out_dir, "pr_curve")


# ============================================================
# CONFUSION MATRICES
# ============================================================

def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    class_names: tuple[str, str] = ("Localized (0)", "Delocalized (1)"),
) -> Dict[str, Path]:
    _ensure_dir(out_dir)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    paths: Dict[str, Path] = {}

    # ---------------------------- COUNTS ----------------------------
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix (Counts)")
    plt.colorbar(im)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    out = _save_plot(fig, out_dir, "confusion_matrix_counts")
    paths["confusion_matrix_counts_png"] = out["png"]
    paths["confusion_matrix_counts_svg"] = out["svg"]

    # --------------------------- NORMALIZED -------------------------
    fig, ax = plt.subplots()
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix (Normalized)")
    plt.colorbar(im)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh_norm = cm_norm.max() / 2.0 if cm_norm.max() > 0 else 0.5
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh_norm else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    out = _save_plot(fig, out_dir, "confusion_matrix_normalized")
    paths["confusion_matrix_normalized_png"] = out["png"]
    paths["confusion_matrix_normalized_svg"] = out["svg"]

    return paths


# ============================================================
# THRESHOLD CURVE
# ============================================================

def plot_threshold_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_dir: Path,
) -> Optional[Dict[str, Path]]:
    if y_proba is None:
        return None

    thresholds = np.linspace(0.05, 0.95, 19)
    f1_scores = [
        f1_score(y_true, (y_proba >= thr).astype(int), zero_division=0)
        for thr in thresholds
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(thresholds, f1_scores, marker="o")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score vs Threshold")
    ax.grid(True)

    return _save_plot(fig, out_dir, "threshold_f1_curve")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(
    model,
    feature_cols: list[str],
    algo: str,
    out_dir: Path,
    top_k: int = 20,
) -> Optional[Dict[str, Path]]:
    importances = None

    # XGB, LGBM, RF, ExtraTrees
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)

    # CatBoost
    if importances is None and hasattr(model, "get_feature_importance"):
        try:
            importances = np.asarray(model.get_feature_importance())
        except Exception:
            importances = None

    if importances is None:
        return None

    _ensure_dir(out_dir)

    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_k]
    top_vals = importances[indices]
    top_feats = [feature_cols[i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(top_feats) + 2))
    y_pos = np.arange(len(top_feats))
    ax.barh(y_pos, top_vals[::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feats[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(top_feats)} Feature Importances ({algo})")

    return _save_plot(fig, out_dir, "feature_importances")

# ============================================================
# DATASET IMBALANCE PLOT
# ============================================================

def plot_dataset_imbalance(
    y_true: np.ndarray,
    out_dir: Path,
) -> Dict[str, Path]:
    """
    Plots the dataset class imbalance: count of 0s vs 1s.
    Saves both PNG and SVG.
    """
    _ensure_dir(out_dir)

    counts = np.bincount(y_true)
    labels = ["Localized (0)", "Delocalized (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, counts, color=["#4C72B0", "#DD8452"])
    ax.set_title("Dataset Class Distribution")
    ax.set_ylabel("Number of Samples")

    # Add text labels above bars
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts) * 0.01, str(v), ha="center", fontsize=10)

    return _save_plot(fig, out_dir, "dataset_imbalance")


# ============================================================
# CREATE ALL PLOTS WRAPPER
# ============================================================

def create_all_plots(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
    y_pred_t05: np.ndarray,
    best_thr: float,
    model,
    feature_cols: list[str],
    best_algo: str,
    model_dir: Path,
) -> Dict[str, Path]:
    """
    Creates all plots and saves both PNG + SVG versions.
    """
    plots_dir = Path(model_dir) / "plots"
    _ensure_dir(plots_dir)

    paths: Dict[str, Path] = {}

    # Confusion matrices @ 0.5
    cm_paths = plot_confusion_matrices(
        y_true=y_true,
        y_pred=y_pred_t05,
        out_dir=plots_dir,
        class_names=("Localized (0)", "Delocalized (1)"),
    )
    paths.update(cm_paths)

    # Dataset imbalance
    imbalance = plot_dataset_imbalance(y_true, plots_dir)
    paths["dataset_imbalance_png"] = imbalance["png"]
    paths["dataset_imbalance_svg"] = imbalance["svg"]

    # ROC curve
    roc = plot_roc_curve(y_true, y_proba, plots_dir)
    if roc:
        paths["roc_curve_png"] = roc["png"]
        paths["roc_curve_svg"] = roc["svg"]

    # PR curve
    pr = plot_pr_curve(y_true, y_proba, plots_dir)
    if pr:
        paths["pr_curve_png"] = pr["png"]
        paths["pr_curve_svg"] = pr["svg"]

    # Threshold vs F1
    thr = plot_threshold_curve(y_true, y_proba, plots_dir)
    if thr:
        paths["threshold_f1_curve_png"] = thr["png"]
        paths["threshold_f1_curve_svg"] = thr["svg"]

    # Feature importance
    fi = plot_feature_importance(
        model=model,
        feature_cols=feature_cols,
        algo=best_algo,
        out_dir=plots_dir,
    )
    if fi:
        paths["feature_importances_png"] = fi["png"]
        paths["feature_importances_svg"] = fi["svg"]

    return paths
