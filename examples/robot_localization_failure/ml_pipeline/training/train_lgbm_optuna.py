import optuna
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from ml_pipeline.utils.paths import DATASETS
from ml_pipeline.utils.common import (
    load_dataset,
    prepare_features,
    fit_scaler,
    apply_scaler,
    compute_metrics,
    print_metrics,
    save_model,
    add_temporal_features,
)

MODEL_NAME = "lightgbm_optuna"
N_TRIALS = 30
USE_TEMPORAL_FEATURES = True
USE_SCANMAP_FEATURES = True
USE_PARTICLE_FEATURES = True
USE_AMCL_POSE = False


def compute_class_weights(y):
    """Compute per-class weights for imbalance handling."""
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    weight_pos = negatives / positives
    return {0: 1.0, 1: weight_pos}


def suggest_params(trial: optuna.Trial) -> dict:
    """Search space for LightGBM."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", -1, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def build_model(params: dict, class_weight):
    """Construct a LightGBM classifier with shared defaults."""
    return lgb.LGBMClassifier(
        **params,
        objective="binary",
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def main():

    # ============================================================
    # 1) Load & prepare dataset
    # ============================================================
    print("Loading training data...")
    df = load_dataset(DATASETS / "train.parquet")

    # ============================================================
    # 1.5) Optional temporal feature generation
    # ============================================================
    if USE_TEMPORAL_FEATURES:
        print("Adding temporal features...")
        df = add_temporal_features(df)
    else:
        print("Skipping temporal features.")

    print("Preparing features...")
    X, y, feature_cols = prepare_features(
        df,
        use_scanmap_features=USE_SCANMAP_FEATURES,
        use_particle_features=USE_PARTICLE_FEATURES,
        use_amcl_pose=USE_AMCL_POSE,
    )

    # ============================================================
    # 2) Train/val split
    # ============================================================
    print("Splitting train/val...")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
        stratify=y,
    )

    # ============================================================
    # 3) Scaling (tree models don't need scaling → scaler = None)
    # ============================================================
    scaler = fit_scaler(X_train, model_type="lgbm")  # returns None
    X_train_scaled = apply_scaler(X_train, scaler)
    X_val_scaled = apply_scaler(X_val, scaler)

    class_weight = compute_class_weights(y_train)

    # ============================================================
    # 4) Optuna search
    # ============================================================
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        model = build_model(params, class_weight)

        model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val_scaled, y_val)],
            eval_metric="binary_logloss",
        )

        y_pred = model.predict(X_val_scaled)
        return f1_score(y_val, y_pred)

    print(f"Starting Optuna search ({N_TRIALS} trials)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = study.best_trial.params
    print(f"Best F1: {study.best_value:.4f}")
    print(f"Best parameters: {best_params}")

    # ============================================================
    # 5) Train best model on full training data
    # ============================================================
    final_class_weight = compute_class_weights(y)
    final_model = build_model(best_params, final_class_weight)

    X_full_scaled = apply_scaler(X, scaler)
    final_model.fit(X_full_scaled, y)

    # ============================================================
    # 6) Evaluate
    # ============================================================
    y_pred = final_model.predict(X_val_scaled)

    print("\n=== Validation Performance (best params) ===")
    metrics = compute_metrics(y_val, y_pred)
    print_metrics(metrics)

    # ============================================================
    # 7) Save model package
    # ============================================================
    save_model(
        model_name=MODEL_NAME,
        model=final_model,
        scaler=scaler,
        feature_cols=feature_cols,
        metrics=metrics,
        extra_metadata={
            "temporal_features": USE_TEMPORAL_FEATURES,
            "model_type": "lightgbm",
            "use_scanmap_features": USE_SCANMAP_FEATURES,
            "use_particle_features": USE_PARTICLE_FEATURES,
            "use_amcl_pose": USE_AMCL_POSE,
            "optuna_best_params": best_params,
            "optuna_best_f1": study.best_value,
            "n_trials": N_TRIALS,
        },
    )


if __name__ == "__main__":
    main()
