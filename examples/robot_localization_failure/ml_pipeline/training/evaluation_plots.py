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


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_dir: Path,
) -> Optional[Path]:
    if y_proba is None:
        return None

    _ensure_dir(out_dir)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    out_path = out_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return out_path


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_dir: Path,
) -> Optional[Path]:
    if y_proba is None:
        return None

    _ensure_dir(out_dir)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    plt.figure()
    plt.plot(recall, precision, label=f"PR curve (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend(loc="lower left")
    out_path = out_dir / "pr_curve.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return out_path


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

    # Raw confusion matrix
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (Counts)")
    plt.colorbar(im)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path_counts = out_dir / "confusion_matrix_counts.png"
    plt.savefig(out_path_counts)
    plt.close()
    paths["confusion_matrix_counts"] = out_path_counts

    # Normalized confusion matrix
    plt.figure()
    im = plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (Normalized)")
    plt.colorbar(im)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh_norm = cm_norm.max() / 2.0 if cm_norm.max() > 0 else 0.5
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh_norm else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path_norm = out_dir / "confusion_matrix_normalized.png"
    plt.savefig(out_path_norm)
    plt.close()
    paths["confusion_matrix_normalized"] = out_path_norm

    return paths


def plot_threshold_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_dir: Path,
) -> Optional[Path]:
    if y_proba is None:
        return None

    _ensure_dir(out_dir)
    thresholds = np.linspace(0.05, 0.95, 19)
    f1_scores = []

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

    plt.figure()
    plt.plot(thresholds, f1_scores, marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.title("F1-score vs. Threshold")
    plt.grid(True)
    out_path = out_dir / "threshold_f1_curve.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return out_path


def plot_feature_importance(
    model,
    feature_cols: list[str],
    algo: str,
    out_dir: Path,
    top_k: int = 20,
) -> Optional[Path]:
    """
    Try to extract feature importances and plot top_k.
    Returns path or None if not supported.
    """
    importances = None

    # XGBoost / LGBM / RF / ExtraTrees
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)

    # CatBoost (explicit API)
    if importances is None and hasattr(model, "get_feature_importance"):
        try:
            importances = np.asarray(model.get_feature_importance())
        except Exception:
            importances = None

    if importances is None:
        return None

    _ensure_dir(out_dir)

    # sort by importance
    indices = np.argsort(importances)[::-1][:top_k]
    top_feats = [feature_cols[i] for i in indices]
    top_vals = importances[indices]

    plt.figure(figsize=(8, 0.4 * len(top_feats) + 2))
    y_pos = np.arange(len(top_feats))
    plt.barh(y_pos, top_vals[::-1])
    plt.yticks(y_pos, top_feats[::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {len(top_feats)} Feature Importances ({algo})")
    plt.tight_layout()
    out_path = out_dir / "feature_importances.png"
    plt.savefig(out_path)
    plt.close()

    return out_path


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
    Create all relevant plots in model_dir / 'plots' and return paths.
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

    # ROC curve
    roc_path = plot_roc_curve(y_true, y_proba, plots_dir)
    if roc_path is not None:
        paths["roc_curve"] = roc_path

    # PR curve
    pr_path = plot_pr_curve(y_true, y_proba, plots_dir)
    if pr_path is not None:
        paths["pr_curve"] = pr_path

    # Threshold vs F1 curve
    thr_path = plot_threshold_curve(y_true, y_proba, plots_dir)
    if thr_path is not None:
        paths["threshold_f1_curve"] = thr_path

    # Feature importance
    fi_path = plot_feature_importance(
        model=model,
        feature_cols=feature_cols,
        algo=best_algo,
        out_dir=plots_dir,
    )
    if fi_path is not None:
        paths["feature_importances"] = fi_path

    return paths
