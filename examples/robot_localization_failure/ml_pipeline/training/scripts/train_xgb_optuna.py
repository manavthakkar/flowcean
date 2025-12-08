import optuna
import xgboost as xgb
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

MODEL_NAME = "xgboost_optuna"
N_TRIALS = 10
USE_TEMPORAL_FEATURES = True
USE_SCANMAP_FEATURES = True
USE_PARTICLE_FEATURES = False
USE_AMCL_POSE = False

def compute_scale_pos_weight(y):
    """Safe scale_pos_weight calculation with imbalance guardrails."""
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return 1.0
    return negatives / positives


def build_model(params: dict, scale_pos_weight: float) -> xgb.XGBClassifier:
    """Construct an XGBoost classifier with shared defaults."""
    return xgb.XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )


def suggest_params(trial: optuna.Trial) -> dict:
    """Search space for Optuna."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.01, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 5.0, log=True),
    }


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
        X, y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
        stratify=y
    )

    # ============================================================
    # 3) Scaling? (NO — XGBoost performs better unscaled)
    # ============================================================
    scaler = fit_scaler(X_train, model_type="xgb")  # returns None
    X_train_scaled = apply_scaler(X_train, scaler)
    X_val_scaled = apply_scaler(X_val, scaler)

    scale_pos_weight = compute_scale_pos_weight(y_train)

    # ============================================================
    # 4) Optuna search
    # ============================================================
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        model = build_model(params, scale_pos_weight=scale_pos_weight)

        model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False,
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
    # 5) Train best model on full training split
    # ============================================================
    full_scale_pos_weight = compute_scale_pos_weight(y)
    final_model = build_model(best_params, scale_pos_weight=full_scale_pos_weight)

    X_full_scaled = apply_scaler(X, scaler)
    final_model.fit(
        X_full_scaled,
        y,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False,
    )

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
            "model_type": "xgboost",
            "use_scanmap_features": USE_SCANMAP_FEATURES,
            "use_particle_features": USE_PARTICLE_FEATURES,
            "use_amcl_pose": USE_AMCL_POSE,
            "optuna_best_params": best_params,
            "optuna_best_f1": study.best_value,
            "n_trials": N_TRIALS,
        }
    )


if __name__ == "__main__":
    main()
