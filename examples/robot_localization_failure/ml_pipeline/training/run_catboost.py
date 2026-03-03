# ml_pipeline/training/train_eval_log_optuna.py

import csv
import datetime

import numpy as np
import optuna
import polars as pl
from catboost import CatBoostClassifier, Pool
from ml_pipeline.training.evaluation_plots import create_all_plots
from ml_pipeline.training.report_generator import generate_pdf_report
from ml_pipeline.training.wandb_logging import (
    finish_wandb_run,
    init_wandb_run,
    log_eval_metrics_to_wandb,
    log_model_artifact,
    log_optuna_summary,
    log_plots_to_wandb,
)
from ml_pipeline.utils.common import (
    LABEL_COL,
    add_temporal_features,
    apply_scaler,
    compute_metrics,
    fit_scaler,
    load_dataset,
    prepare_features,
    print_metrics,
    remove_leaky_columns,
    save_model,
)
from ml_pipeline.utils.paths import ARTIFACTS, DATASETS
from sklearn.metrics import f1_score, precision_score, recall_score

import flowcean.cli

# ====================== CONFIG ==============================

config = flowcean.cli.initialize()

MODEL_NAME = config.optuna.model_name
N_TRIALS = config.optuna.n_trials

USE_TEMPORAL_FEATURES = config.features.use_temporal
USE_SCANMAP_FEATURES = config.features.use_scanmap
USE_PARTICLE_FEATURES = config.features.use_particle
USE_AMCL_POSE = config.features.use_amcl_pose

LOG_PATH = ARTIFACTS / "experiment_log.csv"

TAGS = config.experiment.tags
NOTES = config.experiment.notes


# ============================================================
# CLASS WEIGHTS HELPERS (CatBoost wants list: [w0, w1])
# ============================================================


def compute_class_weights_list(y):
    y = np.asarray(y)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return None
    return [1.0, negatives / positives]


# ============================================================
# HYPERPARAMETER SPACE (CatBoost only) - IMPORTANT:
# use CatBoost-native parameter keys (iterations, depth, etc.)
# ============================================================


def suggest_catboost_params(trial):
    return {
        "iterations": trial.suggest_int("iterations", 300, 1200, step=100),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        ),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", 1e-3, 10.0, log=True
        ),
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature", 0.0, 5.0
        ),
        "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }


# ============================================================
# BUILD MODEL (CatBoost only)
# ============================================================


def build_catboost_model(params, class_weights):
    return CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="F1",
        class_weights=class_weights,  # list [w0, w1] or None
        random_seed=42,
        verbose=False,
    )


# ============================================================
# PREPARE DATASET (single pipeline)
# ============================================================


def prepare_dataset_catboost(df: pl.DataFrame):
    df_cb = df.clone()

    if USE_TEMPORAL_FEATURES:
        df_cb = add_temporal_features(df_cb)

    X, y, feature_cols = prepare_features(
        df_cb,
        use_scanmap_features=USE_SCANMAP_FEATURES,
        use_particle_features=USE_PARTICLE_FEATURES,
        use_amcl_pose=USE_AMCL_POSE,
    )

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
        stratify=y,
    )

    scaler = fit_scaler(X_train, "catboost")
    X_train_scaled = apply_scaler(X_train, scaler)
    X_val_scaled = apply_scaler(X_val, scaler)
    X_full_scaled = apply_scaler(X, scaler)

    cw_train = compute_class_weights_list(y_train)
    cw_full = compute_class_weights_list(y)

    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_full": X_full_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_full": y,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "class_weight_train": cw_train,
        "class_weight_full": cw_full,
    }


# ============================================================
# AUTOMATIC EVALUATION (RETURNS METRICS + y_pred_t05)
# ============================================================


def evaluate_model_automatically(model, scaler, feature_cols, metadata):
    eval_path = DATASETS / "eval.parquet"
    print(f"\nLoading eval dataset: {eval_path}")
    df = pl.read_parquet(eval_path).drop_nulls()

    use_temporal = bool(
        metadata.get("use_temporal_features")
        or metadata.get("temporal_features"),
    )

    if use_temporal:
        print("Adding temporal features for eval dataset...")
        df = add_temporal_features(df)
    else:
        print("ℹModel does NOT use temporal features.")

    df = remove_leaky_columns(df)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in eval dataset: {missing}")

    X = df.select(feature_cols).to_numpy()
    y_true = df[LABEL_COL].to_numpy()

    X_scaled = apply_scaler(X, scaler)

    y_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("\n=== AUTOMATIC EVALUATION (threshold=0.5) ===")
    metrics_t05 = compute_metrics(y_true, y_pred)
    print_metrics(metrics_t05)

    out_path = metadata["model_dir"] / "eval_results.parquet"
    df_out = df.with_columns(
        [
            pl.Series("prediction", y_pred),
            pl.Series("probability", y_proba),
        ]
    )
    df_out.write_parquet(out_path)
    print(f"✔ Saved eval predictions → {out_path}")

    return y_true, y_proba, y_pred, metrics_t05


# ============================================================
# THRESHOLD SWEEP (RETURNS BEST THRESHOLD + METRICS)
# ============================================================


def sweep_thresholds(y_true, y_proba):
    print("\n=== AUTOMATIC THRESHOLD SWEEP ===")
    print("thr\tprec\trec\tF1")

    best_f1 = -1
    best_thr = None
    best_metrics = None

    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = (y_proba >= thr).astype(int)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"{thr:.2f}\t{prec:.3f}\t{rec:.3f}\t{f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_metrics = {"precision": prec, "recall": rec, "f1": f1}

    print(f"\nBest F1 threshold = {best_thr:.2f}  (F1 = {best_f1:.3f})")
    return best_thr, best_metrics


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================


def print_final_summary(n_trials, metrics_t05, best_thr, best_metrics):
    print("\n==================== FINAL MODEL SUMMARY ====================\n")

    print(f"Optuna trials used         : {n_trials}")
    print("Best algorithm             : catboost\n")

    print("Feature Toggles:")
    print(f"    use_temporal_features  : {USE_TEMPORAL_FEATURES}")
    print(f"    use_scanmap_features   : {USE_SCANMAP_FEATURES}")
    print(f"    use_particle_features  : {USE_PARTICLE_FEATURES}")
    print(f"    use_amcl_pose          : {USE_AMCL_POSE}\n")

    print("Performance @ threshold = 0.5")
    print(f"    Precision              : {metrics_t05['precision']:.4f}")
    print(f"    Recall                 : {metrics_t05['recall']:.4f}")
    print(f"    F1-score               : {metrics_t05['f1']:.4f}\n")

    print(f"Best Threshold (F1)        : {best_thr:.2f}")
    print("Performance @ best threshold:")
    print(f"    Precision              : {best_metrics['precision']:.4f}")
    print(f"    Recall                 : {best_metrics['recall']:.4f}")
    print(f"    F1-score               : {best_metrics['f1']:.4f}")

    print("\n==============================================================\n")

    print("\n================ GOOGLE SHEETS EXPORT ================\n")

    sheets_formula = (
        "={"
        f"{metrics_t05['precision']:.4f}, "
        f"{metrics_t05['recall']:.4f}, "
        f"{metrics_t05['f1']:.4f}, "
        f"{best_thr:.2f}, "
        f"{best_metrics['f1']:.4f}, "
        f"{best_metrics['recall']:.4f}, "
        f"{best_metrics['precision']:.4f}"
        "}"
    )

    print(sheets_formula)
    print("\n======================================================\n")


# ============================================================
# APPEND EXPERIMENT LOG
# ============================================================


def append_experiment_log(
    model_name,
    model_dir,
    n_trials,
    metrics_t05,
    best_thr,
    best_metrics,
    train_maps,
    eval_maps,
    odometry,
    notes,
    position_threshold,
    heading_threshold,
):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_PATH.exists()

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "model_name",
                    "model_dir",
                    "optuna_trials",
                    "best_algorithm",
                    "use_temporal",
                    "use_scanmap",
                    "use_particles",
                    "use_amcl_pose",
                    "f1_t05",
                    "precision_t05",
                    "recall_t05",
                    "best_threshold_f1",
                    "f1_best",
                    "recall_best",
                    "precision_best",
                    "train_maps",
                    "eval_maps",
                    "odometry",
                    "notes",
                    "position_threshold",
                    "heading_threshold",
                ]
            )

        writer.writerow(
            [
                model_name,
                str(model_dir),
                n_trials,
                "catboost",
                USE_TEMPORAL_FEATURES,
                USE_SCANMAP_FEATURES,
                USE_PARTICLE_FEATURES,
                USE_AMCL_POSE,
                metrics_t05["f1"],
                metrics_t05["precision"],
                metrics_t05["recall"],
                best_thr,
                best_metrics["f1"],
                best_metrics["recall"],
                best_metrics["precision"],
                ",".join(train_maps) if train_maps else "",
                ",".join(eval_maps) if eval_maps else "",
                odometry,
                notes,
                position_threshold,
                heading_threshold,
            ]
        )


# ============================================================
# MAIN PIPELINE
# ============================================================


def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    wandb_config = {
        "train_maps": config.experiment.train_maps,
        "eval_maps": config.experiment.eval_maps,
        "odometry": config.experiment.odometry,
        "notes": config.experiment.notes,
        "position_threshold": config.localization.position_threshold,
        "heading_threshold": config.localization.heading_threshold,
        "use_temporal": USE_TEMPORAL_FEATURES,
        "use_scanmap": USE_SCANMAP_FEATURES,
        "use_particles": USE_PARTICLE_FEATURES,
        "use_amcl_pose": USE_AMCL_POSE,
        "optuna_trials": N_TRIALS,
        "algorithm": "catboost",
    }

    run = init_wandb_run(
        config_obj=config,
        run_name=MODEL_NAME,
        config_dict=wandb_config,
        tags=TAGS,
        notes=NOTES,
    )

    print("Loading training dataset...")
    df = load_dataset(DATASETS / "train.parquet")

    print("Preparing dataset (CatBoost only)...")
    data = prepare_dataset_catboost(df)

    # -------------------- OPTUNA OBJECTIVE --------------------
    def objective(trial):
        params = suggest_catboost_params(trial)
        model = build_catboost_model(
            params, class_weights=data["class_weight_train"]
        )

        train_pool = Pool(data["X_train"], data["y_train"])
        val_pool = Pool(data["X_val"], data["y_val"])

        model.fit(train_pool, eval_set=val_pool)

        y_pred = model.predict(val_pool)
        y_pred = np.asarray(y_pred).reshape(-1).astype(int)

        return f1_score(
            np.asarray(data["y_val"]).reshape(-1), y_pred, zero_division=0
        )

    print(f"🚀 Starting Optuna search (CatBoost, {N_TRIALS} trials)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = dict(study.best_trial.params)

    print("\nBest Algorithm: catboost")
    print(f"Best Params: {best_params}")
    print(f"Best F1: {study.best_value:.4f}")

    # ---------------- RENAME W&B RUN ----------------
    new_run_name = f"{MODEL_NAME}_catboost_{timestamp}"
    run.name = new_run_name
    run.config.update({"run_name": new_run_name}, allow_val_change=True)
    print(f"Renamed W&B run to: {new_run_name}")

    log_optuna_summary(
        run=run,
        best_algo="catboost",
        best_params=best_params,
        best_f1=study.best_value,
        n_trials=N_TRIALS,
    )

    # -------------------- TRAIN FINAL MODEL --------------------
    final_model = build_catboost_model(
        best_params, class_weights=data["class_weight_full"]
    )

    print("\nTraining best CatBoost model on FULL training data...")
    full_pool = Pool(data["X_full"], data["y_full"])
    final_model.fit(full_pool)

    # Validation metrics (same behavior as before)
    val_pool = Pool(data["X_val"], data["y_val"])
    y_pred_val = final_model.predict(val_pool)
    y_pred_val = np.asarray(y_pred_val).reshape(-1).astype(int)

    print("\n=== VALIDATION METRICS (final model) ===")
    metrics_val = compute_metrics(
        np.asarray(data["y_val"]).reshape(-1), y_pred_val
    )
    print_metrics(metrics_val)

    model_dir = save_model(
        model_name=new_run_name,
        model=final_model,
        scaler=data["scaler"],
        feature_cols=data["feature_cols"],
        metrics=metrics_val,
        add_timestamp=False,
        extra_metadata={
            "selected_algorithm": "catboost",
            "algorithms_considered": ["catboost"],
            "temporal_features": USE_TEMPORAL_FEATURES,
            "use_scanmap_features": USE_SCANMAP_FEATURES,
            "use_particle_features": USE_PARTICLE_FEATURES,
            "use_amcl_pose": USE_AMCL_POSE,
            "optuna_best_params": best_params,
            "optuna_best_f1": study.best_value,
            "n_trials": N_TRIALS,
        },
    )

    # -------------------- AUTOMATIC EVALUATION --------------------
    print("\nAUTOMATIC EVALUATION STARTED...")
    y_true, y_proba, y_pred_t05, metrics_t05 = evaluate_model_automatically(
        final_model,
        data["scaler"],
        data["feature_cols"],
        metadata={
            "model_dir": model_dir,
            "temporal_features": USE_TEMPORAL_FEATURES,
        },
    )

    # -------------------- THRESHOLD SWEEP ----------------------
    print("\nAUTOMATIC THRESHOLD SWEEP STARTED...")
    best_thr, best_metrics = sweep_thresholds(y_true, y_proba)

    # -------------------- PRINT SUMMARY ------------------------
    print_final_summary(
        n_trials=N_TRIALS,
        metrics_t05=metrics_t05,
        best_thr=best_thr,
        best_metrics=best_metrics,
    )

    # -------------------- PLOTS ------------------------
    plot_paths = create_all_plots(
        y_true=y_true,
        y_proba=y_proba,
        y_pred_t05=y_pred_t05,
        best_thr=best_thr,
        model=final_model,
        feature_cols=data["feature_cols"],
        best_algo="catboost",
        model_dir=model_dir,
    )

    # -------------------- REPORT GENERATION ------------------------
    svg_paths = {
        k: v for k, v in plot_paths.items() if str(v).endswith(".svg")
    }

    if config.report.generate_pdf:
        generate_pdf_report(
            model_dir=model_dir,
            model_name=f"{MODEL_NAME}_catboost",
            best_algo="catboost",
            n_trials=N_TRIALS,
            metrics_t05=metrics_t05,
            best_thr=best_thr,
            best_metrics=best_metrics,
            train_maps=config.experiment.train_maps,
            eval_maps=config.experiment.eval_maps,
            odometry=config.experiment.odometry,
            notes=config.experiment.notes,
            feature_flags={
                "temporal": USE_TEMPORAL_FEATURES,
                "scanmap": USE_SCANMAP_FEATURES,
                "particle": USE_PARTICLE_FEATURES,
                "amcl": USE_AMCL_POSE,
            },
            svg_paths=svg_paths,
        )

    # -------------------- W&B LOGGING ------------------------
    log_eval_metrics_to_wandb(
        run=run,
        metrics_t05=metrics_t05,
        best_thr=best_thr,
        best_metrics=best_metrics,
    )

    log_plots_to_wandb(run=run, plot_paths=plot_paths)

    log_model_artifact(
        run=run,
        model_dir=model_dir,
        artifact_name=f"{MODEL_NAME}_catboost_artifact",
    )

    print(
        "\nDONE — CatBoost Training, Evaluation, Plots, and W&B logging finished!\n"
    )

    # -------------------- APPEND EXPERIMENT LOG ----------------
    append_experiment_log(
        model_name=f"{MODEL_NAME}_catboost",
        model_dir=model_dir,
        n_trials=N_TRIALS,
        metrics_t05=metrics_t05,
        best_thr=best_thr,
        best_metrics=best_metrics,
        train_maps=config.experiment.train_maps,
        eval_maps=config.experiment.eval_maps,
        odometry=config.experiment.odometry,
        notes=config.experiment.notes,
        position_threshold=config.localization.position_threshold,
        heading_threshold=config.localization.heading_threshold,
    )

    print(f"Experiment log updated → {LOG_PATH}")

    finish_wandb_run(run)


if __name__ == "__main__":
    main()
