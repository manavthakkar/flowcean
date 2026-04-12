import json
import joblib
import numpy as np
import polars as pl

from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    fbeta_score,
)

from ml_pipeline.utils.paths import MODELS


# ============================================================
#                COLUMN DEFINITIONS
# ============================================================

LABEL_COL = "is_delocalized"

LEAKY_COLUMNS = [
    "time",
    "gt_x", "gt_y", "gt_qx", "gt_qy", "gt_qz", "gt_qw",
    "gt_yaw",
    "position_error", "heading_error_raw", "heading_error",
    "combined_error",
]

# all models must use columns EXCLUDING these
def remove_leaky_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drops columns that leak ground-truth or post-hoc errors."""
    cols_to_drop = [c for c in LEAKY_COLUMNS if c in df.columns]
    if cols_to_drop:
        print(f"Dropping columns: {cols_to_drop}")
    return df.drop(cols_to_drop)


# ============================================================
#                DATA LOADING & PREPARATION
# ============================================================

def load_dataset(parquet_path: Path) -> pl.DataFrame:
    """Reads a parquet and removes null rows."""
    df = pl.read_parquet(parquet_path)
    df = df.drop_nulls()
    return df


def prepare_features(
    df: pl.DataFrame,
    use_scanmap_features: bool = True,
    use_particle_features: bool = True,
    use_amcl_pose: bool = False,
    use_amcl_cov: bool = True,
    use_odom_vel: bool = True,
    use_cmd_vel: bool = True,
    label_col: str = LABEL_COL,
):
    """
    Removes leakage columns, extracts feature matrix X and label y,
    ensures consistent column order.

    All label columns (is_delocalized and lbl_* columns) are dropped from X
    except the one specified by ``label_col``, which becomes y.
    """
    df = remove_leaky_columns(df)

    # Drop non-target label columns (is_delocalized and all lbl_* variants)
    all_label_cols = [c for c in df.columns
                      if c == "is_delocalized" or c.startswith("lbl_")]
    non_target_labels = [c for c in all_label_cols if c != label_col]
    if non_target_labels:
        df = df.drop(non_target_labels)

    # Remove Scan-Map features
    SCANMAP_FEATURES = [
        "point_distance", "point_fitting", "point_inlier", "point_quality",
        "ray_inlier", "ray_inlier_percent", "ray_matching_percent",
        "ray_outlier_percent", "ray_quality",
        "angle_inlier", "angle_quality",
        "line_angle", "line_distance", "line_fitting", "line_length",
    ]

    PARTICLE_FEATURES = [
        "cog_max_distance", "cog_mean_dist", "cog_mean_absolute_deviation",
        "cog_median", "cog_median_absolute_deviation",
        "cog_min_distance", "cog_standard_deviation",
        "circle_radius", "circle_mean", "circle_mean_absolute_deviation",
        "circle_median", "circle_median_absolute_deviation",
        "circle_min_distance", "circle_standard_deviation",
        "num_clusters",
        "main_cluster_variance_x", "main_cluster_variance_y",
    ]

    AMCL_POSE_FEATURES = [
        "amcl_x", "amcl_y",
        "amcl_qx", "amcl_qy", "amcl_qz", "amcl_qw",
        "amcl_yaw",
    ]

    AMCL_COV_FEATURES = ["amcl_cov_x", "amcl_cov_y", "amcl_cov_yaw"]
    ODOM_FEATURES = ["odom_linear_x", "odom_angular_z"]
    CMD_VEL_FEATURES = ["cmd_linear_x", "cmd_angular_z"]

    def drop_feature_set(df: pl.DataFrame, feature_names: list[str], label: str) -> pl.DataFrame:
        """Drop a feature group plus any temporal variants that may have been created."""
        suffixes = ("_diff1", "_mean5", "_std5")
        base_cols = [c for c in feature_names if c in df.columns]
        temporal_cols = [
            f"{c}{s}"
            for c in feature_names
            for s in suffixes
            if f"{c}{s}" in df.columns
        ]
        cols_to_drop = base_cols + temporal_cols
        if cols_to_drop:
            print(f"⚠️  Removing {label} features: {cols_to_drop}")
            df = df.drop(cols_to_drop)
        return df

    if not use_scanmap_features:
        df = drop_feature_set(df, SCANMAP_FEATURES, "Scan-Map")

    if not use_particle_features:
        df = drop_feature_set(df, PARTICLE_FEATURES, "Particle")

    if not use_amcl_pose:
        df = drop_feature_set(df, AMCL_POSE_FEATURES, "AMCL pose")

    if not use_amcl_cov:
        df = drop_feature_set(df, AMCL_COV_FEATURES, "AMCL covariance")

    if not use_odom_vel:
        df = drop_feature_set(df, ODOM_FEATURES, "Odometry velocity")

    if not use_cmd_vel:
        df = drop_feature_set(df, CMD_VEL_FEATURES, "Commanded velocity")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' missing in dataset.")

    y = df[label_col].to_numpy()
    X_df = df.drop([label_col])

    feature_cols = X_df.columns
    X = X_df.to_numpy()

    return X, y, feature_cols


# ============================================================
#                      SCALING
# ============================================================

def fit_scaler(X_train, model_type: str):
    """
    Returns a StandardScaler if the model requires scaling.
    Tree-based models (RF, XGB) do NOT require scaling, so return None.
    """
    from sklearn.preprocessing import StandardScaler

    if model_type in ["rf", "random_forest", "xgb", "xgboost", "tree"]:
        # No scaling needed
        return None

    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(X, scaler):
    """Apply scaler if not None, otherwise return unscaled X."""
    if scaler is None:
        return X
    return scaler.transform(X)


# ============================================================
#                     METRIC UTILITIES
# ============================================================

def compute_metrics(y_true, y_pred):
    """Compute structured performance metrics for saving."""
    metrics = {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "f0.5": float(fbeta_score(y_true,y_pred,beta=0.5)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred),
    }
    return metrics


def print_metrics(metrics):
    """Pretty-print metrics."""
    print("\n=== Evaluation Metrics ===")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1-score:   {metrics['f1']:.4f}")
    print(f"F0.5 score: {metrics['f0.5']:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification Report:")
    print(metrics["classification_report"])


# ============================================================
#                  MODEL SAVE / LOAD
# ============================================================

# ml_pipeline/utils/common.py

import json
import joblib
from datetime import datetime
from ml_pipeline.utils.paths import MODELS
from pathlib import Path


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_model(
    model_name: str,
    model,
    scaler,
    feature_cols,
    metrics,
    extra_metadata: dict | None = None,
    add_timestamp: bool = True,
):
    """
    Saves a complete model package into a UNIQUE timestamped folder:

    artifacts/models/<model_name>_<timestamp>/
        model.pkl
        scaler.pkl
        feature_columns.json
        metrics.json
        metadata.json (optional)

    Parameters
    ----------
    model_name : str
        Name of the model (e.g., "random_forest", "nn", "xgboost").
    model : object
        Trained model instance.
    scaler : object or None
        Optional scaler (e.g., StandardScaler). Saved only if not None.
    feature_cols : list[str]
        List of features used during training.
    metrics : dict
        Validation metrics (precision/recall/F1/etc.)
    extra_metadata : dict or None
        Any additional info such as:
            {"temporal_features": True,
             "model_type": "rf",
             "notes": "..."}
    """

    # -----------------------------------------
    # Create unique timestamped directory
    # -----------------------------------------
    if add_timestamp:
        model_dir_name = f"{model_name}_{timestamp()}"
    else:
        model_dir_name = model_name
    model_dir = MODELS / model_dir_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    # Save model
    # -----------------------------------------
    joblib.dump(model, model_dir / "model.pkl")

    # Save scaler (if exists)
    if scaler is not None:
        joblib.dump(scaler, model_dir / "scaler.pkl")

    # Save feature column list
    with open(model_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save validation metrics
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # -----------------------------------------
    # Save extra metadata (if provided)
    # -----------------------------------------
    if extra_metadata is not None:
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(extra_metadata, f, indent=2)

    print(f"\n✔ Model saved in: {model_dir}")
    return model_dir

def load_model(model_name: str):
    """
    Loads: model, scaler (maybe None), and feature columns
    """
    model_dir = MODELS / model_name
    model = joblib.load(model_dir / "model.pkl")

    scaler_path = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    with open(model_dir / "feature_columns.json", "r") as f:
        feature_cols = json.load(f)

    return model, scaler, feature_cols


# ============================================================
#                      PREDICTION
# ============================================================

def predict_with_model(model, scaler, X):
    """Apply scaling if needed and run inference."""
    X_scaled = apply_scaler(X, scaler)
    return model.predict(X_scaled), model.predict_proba(X_scaled)[:, 1]

RESET_PRE_ERROR_MIN = 0.5      # fallback combined_error: prev must exceed this
RESET_POST_ERROR_MAX = 0.1     # fallback combined_error: curr must be below this
RESET_MIN_DROP = 0.4           # fallback combined_error: required drop (prev - curr)
POSITION_PRE_ERROR_MIN = 0.5   # position_error: prev must exceed this
POSITION_POST_ERROR_MAX = 0.1  # position_error: curr must be below this
POSITION_MIN_DROP = 0.4        # position_error: required drop (prev - curr)
HEADING_PRE_ERROR_MIN = 0.5    # |heading_error|: prev must exceed this
HEADING_POST_ERROR_MAX = 0.1   # |heading_error|: curr must be below this
HEADING_MIN_DROP = 0.4         # |heading_error|: required drop (prev - curr)
RESET_POST_WINDOW_SAMPLES = 10 # samples to drop after each reset (scaled by median timestep)
TEMPORAL_ROLLING_WINDOW = 5    # rolling window size; ensures we drop at least this many samples


def detect_amcl_resets(df: pl.DataFrame, max_gap: float | None = None):
    """
    Identify AMCL reset points using existing error signals.

    A reset is detected when either position_error or heading_error (absolute)
    sharply drops from a high value to a low value:
      - prev error > *_PRE_ERROR_MIN
      - curr error < *_POST_ERROR_MAX
      - drop magnitude > *_MIN_DROP
    If neither position_error nor heading_error is present, combined_error is
    used as a fallback with the RESET_* thresholds.
    """
    has_position = "position_error" in df.columns
    has_heading = "heading_error" in df.columns
    has_combined = "combined_error" in df.columns
    if not (has_position or has_heading or has_combined):
        return []

    df = df.sort("time")
    gap_expr = pl.lit(True)
    if max_gap is not None and max_gap > 0:
        gap_expr = (pl.col("time") - pl.col("time").shift(1)) <= max_gap

    reset_exprs: list[pl.Expr] = []

    if has_position:
        pos = pl.col("position_error")
        reset_exprs.append(
            (pos.shift(1) > POSITION_PRE_ERROR_MIN) &
            (pos < POSITION_POST_ERROR_MAX) &
            ((pos.shift(1) - pos) > POSITION_MIN_DROP) &
            gap_expr
        )

    if has_heading:
        head = pl.col("heading_error").abs()
        reset_exprs.append(
            (head.shift(1) > HEADING_PRE_ERROR_MIN) &
            (head < HEADING_POST_ERROR_MAX) &
            ((head.shift(1) - head) > HEADING_MIN_DROP) &
            gap_expr
        )

    # Fallback to combined_error only if neither position nor heading is present
    if not reset_exprs and has_combined:
        err = pl.col("combined_error")
        reset_exprs.append(
            (err.shift(1) > RESET_PRE_ERROR_MIN) &
            (err < RESET_POST_ERROR_MAX) &
            ((err.shift(1) - err) > RESET_MIN_DROP) &
            gap_expr
        )

    reset_expr = pl.any_horizontal(reset_exprs)
    return df.filter(reset_expr)["time"].to_list()


def remove_post_reset_artifacts(
    df: pl.DataFrame,
    samples_to_skip: int = RESET_POST_WINDOW_SAMPLES,
):
    """
    Drop a short window of samples immediately after each detected AMCL reset.

    Returns the cleaned dataframe plus metadata about the removed window.
    """
    df = df.sort("time")
    # Estimate the duration to drop using the median timestep
    median_step = df["time"].diff().drop_nulls().median()
    median_step = float(median_step) if median_step is not None else 0.0

    max_gap = median_step * samples_to_skip if median_step > 0 else None
    reset_times = detect_amcl_resets(df, max_gap=max_gap)
    window_size = int(median_step * samples_to_skip) if median_step > 0 else 0

    drop_expr = pl.lit(False)
    if reset_times:
        # Drop the reset row itself and the next `samples_to_skip` rows.
        # Use a left-closed / right-open window so we do not over-drop by one.
        drop_windows = [
            pl.col("time").is_between(t, t + window_size, closed="left")
            for t in reset_times
        ]
        drop_expr = pl.any_horizontal(drop_windows)

    cleaned = df.filter(~drop_expr)
    removed_rows = df.height - cleaned.height

    return cleaned, {
        "reset_times": reset_times,
        "median_step": median_step,
        "window_size": window_size,
        "window_seconds": window_size / 1e9 if window_size else 0.0,
        "samples_to_skip": samples_to_skip,
        "rows_removed": removed_rows,
    }

def add_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add simple temporal / rolling features to the dataset.

    For each numeric base feature (excluding time, labels, GT, and error columns)
    this adds:
      - <col>_diff1   : first difference (col(t) - col(t-1))
      - <col>_mean5   : rolling mean over window=5
      - <col>_std5    : rolling std  over window=5

    The frame is first sorted by 'time', temporal statistics are computed,
    and then a short post-reset window is removed to ensure rows affected by
    AMCL resets do not remain in the dataset.
    """
    if "time" not in df.columns:
        raise ValueError("Expected a 'time' column in the dataset for temporal features.")

    # Sort by time
    df = df.sort("time")

    # Columns that must NOT be used as base features
    exclude_cols = {
        "time",
        "is_delocalized",
        "gt_x", "gt_y", "gt_qx", "gt_qy", "gt_qz", "gt_qw",
        "gt_yaw",
        "position_error", "heading_error_raw", "heading_error",
        "combined_error",
    }

    # Select numeric columns that are not excluded (also skip lbl_* label columns)
    numeric_types = (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    schema = df.schema

    base_features: list[str] = [
        name
        for name, dtype in schema.items()
        if name not in exclude_cols
        and not name.startswith("lbl_")
        and isinstance(dtype, numeric_types)
    ]

    print(f"[add_temporal_features] Base features ({len(base_features)}): {base_features}")

    # For each base feature, add diff1, mean5, std5
    new_cols: list[pl.Expr] = []
    for col in base_features:
        new_cols.extend([
            # First difference
            (pl.col(col) - pl.col(col).shift(1)).alias(f"{col}_diff1"),
            # Rolling mean over window=5 (min_periods=1 so early rows are still valid)
            pl.col(col)
            .rolling_mean(window_size=5, min_samples=1)
            .alias(f"{col}_mean5"),
            # Rolling std over window=5
            pl.col(col)
            .rolling_std(window_size=5, min_samples=1)
            .alias(f"{col}_std5"),
        ])

    if not new_cols:
        print("[add_temporal_features] No temporal features created (no suitable columns).")
        return df

    df = df.with_columns(new_cols)

    # Remove post-reset windows after temporal features to ensure no temporal
    # columns are contaminated by reset jumps. Drop enough rows to cover the
    # largest temporal window (rolling window size).
    post_reset_skip = max(RESET_POST_WINDOW_SAMPLES, TEMPORAL_ROLLING_WINDOW)
    df, reset_meta = remove_post_reset_artifacts(df, samples_to_skip=post_reset_skip)
    if reset_meta["reset_times"]:
        print(
            f"[add_temporal_features] Detected {len(reset_meta['reset_times'])} AMCL reset(s); "
            f"removed {reset_meta['rows_removed']} row(s) "
            f"({reset_meta['samples_to_skip']} samples ≈ "
            f"{reset_meta['window_seconds']:.2f}s) after computing temporal features."
        )

    print(f"[add_temporal_features] Added {len(new_cols)} new temporal feature columns.")
    # Shift/rolling ops introduce nulls in the first row; drop them so downstream
    # models that disallow NaN (e.g., LogisticRegression) can train without errors.
    df = df.drop_nulls()
    return df
