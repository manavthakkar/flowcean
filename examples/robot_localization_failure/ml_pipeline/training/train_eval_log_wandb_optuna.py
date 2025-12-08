# ml_pipeline/training/train_eval_log_optuna.py

import optuna
import lightgbm as lgb
import xgboost as xgb
import flowcean.cli
import polars as pl
import numpy as np
import csv

from catboost import CatBoostClassifier, Pool
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from ml_pipeline.utils.paths import DATASETS, ARTIFACTS
from ml_pipeline.utils.common import (
    add_temporal_features,
    apply_scaler,
    compute_metrics,
    fit_scaler,
    load_dataset,
    prepare_features,
    remove_leaky_columns,
    print_metrics,
    save_model,
    LABEL_COL,
)

from ml_pipeline.training.evaluation_plots import create_all_plots
from ml_pipeline.training.wandb_logging import (
    init_wandb_run,
    log_optuna_summary,
    log_eval_metrics_to_wandb,
    log_plots_to_wandb,
    log_model_artifact,
    finish_wandb_run,
)

# ====================== CONFIG ==============================

config = flowcean.cli.initialize()

MODEL_NAME = config.optuna.model_name
N_TRIALS = config.optuna.n_trials

USE_TEMPORAL_FEATURES = config.features.use_temporal
USE_SCANMAP_FEATURES = config.features.use_scanmap
USE_PARTICLE_FEATURES = config.features.use_particle
USE_AMCL_POSE = config.features.use_amcl_pose

ALGORITHMS = {
    "lgbm": {"model_type": "lgbm"},
    "rf": {"model_type": "rf"},
    "extratrees": {"model_type": "extratrees"},
    "catboost": {"model_type": "catboost"},
    "xgb": {"model_type": "xgb"},
}

LOG_PATH = ARTIFACTS / "experiment_log.csv"


# ============================================================
# CLASS WEIGHTS / SCALE POS WEIGHT HELPERS
# ============================================================

def compute_class_weights_dict(y):
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    return {0: 1.0, 1: negatives / positives}


def compute_class_weights_list(y):
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    return [1.0, negatives / positives]


def compute_scale_pos_weight(y):
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0:
        return 1.0
    return negatives / positives


# ============================================================
# HYPERPARAMETER SPACES
# ============================================================

def suggest_params(trial, algo):
    pfx = f"{algo}_"

    if algo == "lgbm":
        return {
            "n_estimators": trial.suggest_int(f"{pfx}n_estimators", 200, 900, step=50),
            "learning_rate": trial.suggest_float(f"{pfx}learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int(f"{pfx}num_leaves", 16, 256),
            "max_depth": trial.suggest_int(f"{pfx}max_depth", -1, 16),
            "min_child_samples": trial.suggest_int(f"{pfx}min_child_samples", 5, 50),
            "subsample": trial.suggest_float(f"{pfx}subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(f"{pfx}colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float(f"{pfx}reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float(f"{pfx}reg_lambda", 1e-3, 10.0, log=True),
        }

    if algo == "rf":
        return {
            "n_estimators": trial.suggest_int(f"{pfx}n_estimators", 200, 900, step=50),
            "max_depth": trial.suggest_int(f"{pfx}max_depth", 5, 30),
            "min_samples_split": trial.suggest_int(f"{pfx}min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int(f"{pfx}min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical(f"{pfx}max_features", ["sqrt", "log2", 0.5, 0.8, 1.0]),
            "bootstrap": trial.suggest_categorical(f"{pfx}bootstrap", [True, False]),
        }

    if algo == "extratrees":
        return {
            "n_estimators": trial.suggest_int(f"{pfx}n_estimators", 200, 900, step=50),
            "max_depth": trial.suggest_int(f"{pfx}max_depth", 5, 40),
            "min_samples_split": trial.suggest_int(f"{pfx}min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int(f"{pfx}min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical(f"{pfx}max_features", ["sqrt", "log2", 0.5, 0.8, 1.0]),
            "bootstrap": trial.suggest_categorical(f"{pfx}bootstrap", [True, False]),
        }

    if algo == "catboost":
        return {
            "iterations": trial.suggest_int(f"{pfx}iterations", 300, 1200, step=100),
            "depth": trial.suggest_int(f"{pfx}depth", 4, 10),
            "learning_rate": trial.suggest_float(f"{pfx}learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float(f"{pfx}l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float(f"{pfx}bagging_temperature", 0.0, 5.0),
            "random_strength": trial.suggest_float(f"{pfx}random_strength", 0.5, 5.0),
            "subsample": trial.suggest_float(f"{pfx}subsample", 0.6, 1.0),
            "border_count": trial.suggest_int(f"{pfx}border_count", 32, 255),
        }

    if algo == "xgb":
        return {
            "n_estimators": trial.suggest_int(f"{pfx}n_estimators", 200, 900, step=50),
            "max_depth": trial.suggest_int(f"{pfx}max_depth", 3, 10),
            "learning_rate": trial.suggest_float(f"{pfx}learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float(f"{pfx}subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(f"{pfx}colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float(f"{pfx}min_child_weight", 0.01, 10.0, log=True),
            "gamma": trial.suggest_float(f"{pfx}gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float(f"{pfx}reg_lambda", 0.001, 10.0, log=True),
            "reg_alpha": trial.suggest_float(f"{pfx}reg_alpha", 0.001, 5.0, log=True),
        }

    raise ValueError(f"Unsupported algorithm: {algo}")


# ============================================================
# BUILD MODEL FROM PARAMETERS
# ============================================================

def build_model(algo, params, class_weight, scale_pos_weight):
    if algo == "lgbm":
        return lgb.LGBMClassifier(
            **params, objective="binary", class_weight=class_weight, random_state=42, n_jobs=-1, verbose=-1
        )

    if algo == "rf":
        return RandomForestClassifier(
            **params, n_jobs=-1, class_weight=class_weight, random_state=42
        )

    if algo == "extratrees":
        return ExtraTreesClassifier(
            **params, n_jobs=-1, class_weight=class_weight, random_state=42
        )

    if algo == "catboost":
        return CatBoostClassifier(
            **params, loss_function="Logloss", eval_metric="F1",
            class_weights=class_weight, random_seed=42, verbose=False
        )

    if algo == "xgb":
        return xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )

    raise ValueError(f"Unsupported algorithm: {algo}")


# ============================================================
# STRIP PREFIX
# ============================================================

def extract_algo_params(all_params, algo):
    prefix = f"{algo}_"
    return {k.replace(prefix, ""): v for k, v in all_params.items() if k.startswith(prefix)}


# ============================================================
# PREPARE DATASETS
# ============================================================

def prepare_datasets(df):
    prepared = {}

    for algo, cfg in ALGORITHMS.items():
        df_algo = df.clone()

        if USE_TEMPORAL_FEATURES:
            df_algo = add_temporal_features(df_algo)

        X, y, feature_cols = prepare_features(
            df_algo,
            use_scanmap_features=USE_SCANMAP_FEATURES,
            use_particle_features=USE_PARTICLE_FEATURES,
            use_amcl_pose=USE_AMCL_POSE,
        )

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
        )

        scaler = fit_scaler(X_train, cfg["model_type"])
        X_train_scaled = apply_scaler(X_train, scaler)
        X_val_scaled = apply_scaler(X_val, scaler)
        X_full_scaled = apply_scaler(X, scaler)

        if algo == "catboost":
            cw_train = compute_class_weights_list(y_train)
            cw_full = compute_class_weights_list(y)
            spw_train = None
            spw_full = None
        elif algo == "xgb":
            cw_train = None
            cw_full = None
            spw_train = compute_scale_pos_weight(y_train)
            spw_full = compute_scale_pos_weight(y)
        else:
            cw_train = compute_class_weights_dict(y_train)
            cw_full = compute_class_weights_dict(y)
            spw_train = None
            spw_full = None

        prepared[algo] = {
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
            "scale_train": spw_train,
            "scale_full": spw_full,
        }

    return prepared


# ============================================================
# AUTOMATIC EVALUATION (RETURNS METRICS + y_pred_t05)
# ============================================================

def evaluate_model_automatically(model, scaler, feature_cols, metadata):
    eval_path = DATASETS / "eval.parquet"
    print(f"\n📘 Loading eval dataset: {eval_path}")
    df = pl.read_parquet(eval_path).drop_nulls()

    use_temporal = bool(
        metadata.get("use_temporal_features") or metadata.get("temporal_features")
    )

    if use_temporal:
        print("🔧 Adding temporal features for eval dataset...")
        df = add_temporal_features(df)
    else:
        print("ℹ️ Model does NOT use temporal features.")

    df = remove_leaky_columns(df)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing columns in eval dataset: {missing}")

    X = df.select(feature_cols).to_numpy()
    y_true = df[LABEL_COL].to_numpy()

    X_scaled = apply_scaler(X, scaler)

    # predictions @ thr=0.5
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_scaled)
        y_proba = None

    print("\n=== AUTOMATIC EVALUATION (threshold=0.5) ===")
    metrics_t05 = compute_metrics(y_true, y_pred)
    print_metrics(metrics_t05)

    # save predictions
    out_path = metadata["model_dir"] / "eval_results.parquet"
    df_out = df.with_columns([
        pl.Series("prediction", y_pred),
        pl.Series("probability", y_proba if y_proba is not None else [None] * len(y_pred))
    ])
    df_out.write_parquet(out_path)
    print(f"✔ Saved eval predictions → {out_path}")

    return y_true, y_proba, y_pred, metrics_t05


# ============================================================
# THRESHOLD SWEEP (RETURNS BEST THRESHOLD + METRICS)
# ============================================================

def sweep_thresholds(y_true, y_proba):
    if y_proba is None:
        print("⚠️ Model has no predict_proba → skipping threshold sweep.")
        return None, None

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

    print(f"\n🏆 Best F1 threshold = {best_thr:.2f}  (F1 = {best_f1:.3f})")

    return best_thr, best_metrics


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_final_summary(
    n_trials,
    best_algo,
    metrics_t05,
    best_thr,
    best_metrics
):
    print("\n==================== FINAL MODEL SUMMARY ====================\n")

    print(f"Optuna trials used         : {n_trials}")
    print(f"Best algorithm             : {best_algo}\n")

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


# ============================================================
# APPEND EXPERIMENT LOG
# ============================================================

def append_experiment_log(
    model_name,
    model_dir,
    n_trials,
    best_algo,
    metrics_t05,
    best_thr,
    best_metrics,
    train_maps,
    eval_maps,
    odometry,
    notes,
    position_threshold,
    heading_threshold
):
    # Ensure folder exists
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_exists = LOG_PATH.exists()

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
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
                "heading_threshold"
            ])

        writer.writerow([
            model_name,
            str(model_dir),
            n_trials,
            best_algo,
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
            heading_threshold
        ])


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    # -------------------- INIT W&B --------------------
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
    }
    run = init_wandb_run(
        config_obj=config,
        run_name=MODEL_NAME,
        config_dict=wandb_config,
    )

    print("📘 Loading training dataset...")
    df = load_dataset(DATASETS / "train.parquet")

    print("📘 Preparing datasets per algorithm...")
    prepared = prepare_datasets(df)

    # -------------------- OPTUNA OBJECTIVE --------------------
    def objective(trial):
        algo = trial.suggest_categorical("algorithm", list(ALGORITHMS.keys()))
        data = prepared[algo]

        params = suggest_params(trial, algo)
        model = build_model(
            algo,
            params,
            class_weight=data["class_weight_train"],
            scale_pos_weight=data["scale_train"]
        )

        if algo == "catboost":
            train_pool = Pool(data["X_train"], data["y_train"])
            val_pool = Pool(data["X_val"], data["y_val"])
            model.fit(train_pool, eval_set=val_pool)
            y_pred = model.predict(val_pool)
        else:
            model.fit(data["X_train"], data["y_train"])
            y_pred = model.predict(data["X_val"])

        return f1_score(data["y_val"], y_pred)

    print(f"🚀 Starting Optuna search ({N_TRIALS} trials)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_algo = study.best_trial.params["algorithm"]
    best_params = extract_algo_params(study.best_trial.params, best_algo)

    print(f"\n🏆 Best Algorithm: {best_algo}")
    print(f"🏆 Best Params: {best_params}")
    print(f"🏆 Best F1: {study.best_value:.4f}")

    # W&B: log Optuna summary
    log_optuna_summary(
        run=run,
        best_algo=best_algo,
        best_params=best_params,
        best_f1=study.best_value,
        n_trials=N_TRIALS,
    )

    # -------------------- TRAIN FINAL MODEL --------------------

    data = prepared[best_algo]
    final_model = build_model(
        best_algo,
        best_params,
        class_weight=data["class_weight_full"],
        scale_pos_weight=data["scale_full"]
    )

    print("\n📘 Training best model on FULL training data...")
    if best_algo == "catboost":
        full_pool = Pool(data["X_full"], data["y_full"])
        final_model.fit(full_pool)
        val_pool = Pool(data["X_val"], data["y_val"])
        y_pred_val = final_model.predict(val_pool)
    else:
        final_model.fit(data["X_full"], data["y_full"])
        y_pred_val = final_model.predict(data["X_val"])

    print("\n=== VALIDATION METRICS (final model) ===")
    metrics_val = compute_metrics(data["y_val"], y_pred_val)
    print_metrics(metrics_val)

    model_dir = save_model(
        model_name=f"{MODEL_NAME}_{best_algo}",
        model=final_model,
        scaler=data["scaler"],
        feature_cols=data["feature_cols"],
        metrics=metrics_val,
        extra_metadata={
            "selected_algorithm": best_algo,
            "algorithms_considered": list(ALGORITHMS.keys()),
            "temporal_features": USE_TEMPORAL_FEATURES,
            "use_scanmap_features": USE_SCANMAP_FEATURES,
            "use_particle_features": USE_PARTICLE_FEATURES,
            "use_amcl_pose": USE_AMCL_POSE,
            "optuna_best_params": best_params,
            "optuna_best_f1": study.best_value,
            "n_trials": N_TRIALS,
        }
    )

    # -------------------- AUTOMATIC EVALUATION --------------------

    print("\n📘 AUTOMATIC EVALUATION STARTED...")
    y_true, y_proba, y_pred_t05, metrics_t05 = evaluate_model_automatically(
        final_model,
        data["scaler"],
        data["feature_cols"],
        metadata={"model_dir": model_dir, "temporal_features": USE_TEMPORAL_FEATURES}
    )

    # -------------------- THRESHOLD SWEEP ----------------------

    print("\n📘 AUTOMATIC THRESHOLD SWEEP STARTED...")
    best_thr, best_metrics = sweep_thresholds(y_true, y_proba)

    # -------------------- PRINT SUMMARY ------------------------

    print_final_summary(
        n_trials=N_TRIALS,
        best_algo=best_algo,
        metrics_t05=metrics_t05,
        best_thr=best_thr,
        best_metrics=best_metrics
    )

    # -------------------- PLOTS ------------------------

    plot_paths = create_all_plots(
        y_true=y_true,
        y_proba=y_proba,
        y_pred_t05=y_pred_t05,
        best_thr=best_thr,
        model=final_model,
        feature_cols=data["feature_cols"],
        best_algo=best_algo,
        model_dir=model_dir,
    )

    # -------------------- W&B METRICS & PLOTS & ARTIFACTS ------------------------

    log_eval_metrics_to_wandb(
        run=run,
        metrics_t05=metrics_t05,
        best_thr=best_thr,
        best_metrics=best_metrics,
    )

    log_plots_to_wandb(
        run=run,
        plot_paths=plot_paths,
    )

    log_model_artifact(
        run=run,
        model_dir=model_dir,
        artifact_name=f"{MODEL_NAME}_{best_algo}_artifact",
    )

    print("\n🎉 DONE — Training, Evaluation, Plots, and W&B logging finished!\n")

    # -------------------- APPEND EXPERIMENT LOG ----------------

    append_experiment_log(
        model_name=f"{MODEL_NAME}_{best_algo}",
        model_dir=model_dir,
        n_trials=N_TRIALS,
        best_algo=best_algo,
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

    print(f"✔ Experiment log updated → {LOG_PATH}")

    # -------------------- FINISH W&B ------------------------
    finish_wandb_run(run)


if __name__ == "__main__":
    main()
