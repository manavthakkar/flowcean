import optuna
from catboost import CatBoostClassifier, Pool
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

MODEL_NAME = "catboost_optuna"
N_TRIALS = 10
USE_TEMPORAL_FEATURES = True
USE_SCANMAP_FEATURES = True
USE_PARTICLE_FEATURES = False
USE_AMCL_POSE = False


def compute_class_weights(y):
    """Compute per-class weights for imbalance handling."""
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    weight_pos = negatives / positives
    return [1.0, weight_pos]


def suggest_params(trial: optuna.Trial) -> dict:
    """Search space for CatBoost."""
    return {
        "iterations": trial.suggest_int("iterations", 300, 1200, step=100),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }


def build_model(params: dict, class_weights):
    """Construct a CatBoost classifier with shared defaults."""
    return CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="F1",
        class_weights=class_weights,
        random_seed=42,
        verbose=False,
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
    # 3) Scaling (CatBoost handles raw features → scaler = None)
    # ============================================================
    scaler = fit_scaler(X_train, model_type="catboost")  # returns None
    X_train_scaled = apply_scaler(X_train, scaler)
    X_val_scaled = apply_scaler(X_val, scaler)

    class_weights = compute_class_weights(y_train)

    # ============================================================
    # 4) Optuna search
    # ============================================================
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        model = build_model(params, class_weights)

        train_pool = Pool(X_train_scaled, y_train)
        val_pool = Pool(X_val_scaled, y_val)

        model.fit(train_pool, eval_set=val_pool)

        y_pred = model.predict(val_pool)
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
    final_class_weights = compute_class_weights(y)
    final_model = build_model(best_params, final_class_weights)

    X_full_scaled = apply_scaler(X, scaler)
    full_pool = Pool(X_full_scaled, y)
    final_model.fit(full_pool)

    # ============================================================
    # 6) Evaluate
    # ============================================================
    val_pool = Pool(X_val_scaled, y_val)
    y_pred = final_model.predict(val_pool)

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
            "model_type": "catboost",
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
