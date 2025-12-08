import optuna
import lightgbm as lgb
import xgboost as xgb
import flowcean.cli
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from ml_pipeline.utils.paths import DATASETS
from ml_pipeline.utils.common import (
    add_temporal_features,
    apply_scaler,
    compute_metrics,
    fit_scaler,
    load_dataset,
    prepare_features,
    print_metrics,
    save_model,
)

config = flowcean.cli.initialize()

MODEL_NAME = config.optuna.model_name
N_TRIALS = config.optuna.n_trials

# Global feature toggles (applied to every algorithm)
USE_TEMPORAL_FEATURES = config.features.use_temporal
USE_SCANMAP_FEATURES = config.features.use_scanmap
USE_PARTICLE_FEATURES = config.features.use_particle
USE_AMCL_POSE = config.features.use_amcl_pose

# Model-type mapping for scaler handling
ALGORITHMS = {
    "lgbm": {"model_type": "lgbm"},
    "rf": {"model_type": "rf"},
    "extratrees": {"model_type": "extratrees"},
    "catboost": {"model_type": "catboost"},
    "xgb": {"model_type": "xgb"},
}


def compute_class_weights_dict(y):
    """Compute per-class weights for tree ensembles."""
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    return {0: 1.0, 1: negatives / positives}


def compute_class_weights_list(y):
    """CatBoost expects weights as a list instead of a dict."""
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    return [1.0, negatives / positives]


def compute_scale_pos_weight(y):
    """Safe scale_pos_weight for XGBoost."""
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return 1.0
    return negatives / positives


def suggest_params(trial: optuna.Trial, algo: str) -> dict:
    """Algorithm-specific hyperparameter search spaces (prefixed for Optuna)."""
    pfx = f"{algo}_"

    if algo == "lgbm":
        return {
            "n_estimators": trial.suggest_int(f"{pfx}n_estimators", 200, 900, step=50),
            "learning_rate": trial.suggest_float(
                f"{pfx}learning_rate", 0.01, 0.3, log=True
            ),
            "num_leaves": trial.suggest_int(f"{pfx}num_leaves", 16, 256),
            "max_depth": trial.suggest_int(f"{pfx}max_depth", -1, 16),
            "min_child_samples": trial.suggest_int(
                f"{pfx}min_child_samples", 5, 50
            ),
            "subsample": trial.suggest_float(f"{pfx}subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                f"{pfx}colsample_bytree", 0.6, 1.0
            ),
            "reg_alpha": trial.suggest_float(f"{pfx}reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float(f"{pfx}reg_lambda", 1e-3, 10.0, log=True),
        }

    if algo == "rf":
        return {
            "n_estimators": trial.suggest_int(f"{pfx}n_estimators", 200, 900, step=50),
            "max_depth": trial.suggest_int(f"{pfx}max_depth", 5, 30),
            "min_samples_split": trial.suggest_int(f"{pfx}min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int(f"{pfx}min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical(
                f"{pfx}max_features", ["sqrt", "log2", 0.5, 0.8, 1.0]
            ),
            "bootstrap": trial.suggest_categorical(f"{pfx}bootstrap", [True, False]),
        }

    if algo == "extratrees":
        return {
            "n_estimators": trial.suggest_int(f"{pfx}n_estimators", 200, 900, step=50),
            "max_depth": trial.suggest_int(f"{pfx}max_depth", 5, 40),
            "min_samples_split": trial.suggest_int(f"{pfx}min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int(f"{pfx}min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical(
                f"{pfx}max_features", ["sqrt", "log2", 0.5, 0.8, 1.0]
            ),
            "bootstrap": trial.suggest_categorical(f"{pfx}bootstrap", [True, False]),
        }

    if algo == "catboost":
        return {
            "iterations": trial.suggest_int(f"{pfx}iterations", 300, 1200, step=100),
            "depth": trial.suggest_int(f"{pfx}depth", 4, 10),
            "learning_rate": trial.suggest_float(
                f"{pfx}learning_rate", 0.01, 0.3, log=True
            ),
            "l2_leaf_reg": trial.suggest_float(
                f"{pfx}l2_leaf_reg", 1e-3, 10.0, log=True
            ),
            "bagging_temperature": trial.suggest_float(
                f"{pfx}bagging_temperature", 0.0, 5.0
            ),
            "random_strength": trial.suggest_float(
                f"{pfx}random_strength", 0.5, 5.0
            ),
            "subsample": trial.suggest_float(f"{pfx}subsample", 0.6, 1.0),
            "border_count": trial.suggest_int(f"{pfx}border_count", 32, 255),
        }

    if algo == "xgb":
        return {
            "n_estimators": trial.suggest_int(f"{pfx}n_estimators", 200, 900, step=50),
            "max_depth": trial.suggest_int(f"{pfx}max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                f"{pfx}learning_rate", 0.01, 0.3, log=True
            ),
            "subsample": trial.suggest_float(f"{pfx}subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                f"{pfx}colsample_bytree", 0.6, 1.0
            ),
            "min_child_weight": trial.suggest_float(
                f"{pfx}min_child_weight", 0.01, 10.0, log=True
            ),
            "gamma": trial.suggest_float(f"{pfx}gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float(
                f"{pfx}reg_lambda", 0.001, 10.0, log=True
            ),
            "reg_alpha": trial.suggest_float(
                f"{pfx}reg_alpha", 0.001, 5.0, log=True
            ),
        }

    raise ValueError(f"Unsupported algorithm: {algo}")


def build_model(algo: str, params: dict, class_weight, scale_pos_weight):
    if algo == "lgbm":
        return lgb.LGBMClassifier(
            **params,
            objective="binary",
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    if algo == "rf":
        return RandomForestClassifier(
            **params,
            n_jobs=-1,
            class_weight=class_weight,
            random_state=42,
        )
    if algo == "extratrees":
        return ExtraTreesClassifier(
            **params,
            n_jobs=-1,
            class_weight=class_weight,
            random_state=42,
        )
    if algo == "catboost":
        return CatBoostClassifier(
            **params,
            loss_function="Logloss",
            eval_metric="F1",
            class_weights=class_weight,
            random_seed=42,
            verbose=False,
        )
    if algo == "xgb":
        return xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        )
    raise ValueError(f"Unsupported algorithm: {algo}")


def extract_algo_params(trial_params: dict, algo: str) -> dict:
    """Strip the algorithm prefix from the best trial's parameters."""
    prefix = f"{algo}_"
    return {k.replace(prefix, ""): v for k, v in trial_params.items() if k.startswith(prefix)}


def prepare_datasets(df):
    """Precompute feature sets, splits, and scalers for each algorithm."""
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

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=True,
            random_state=42,
            stratify=y,
        )

        scaler = fit_scaler(X_train, model_type=cfg["model_type"])
        X_train_scaled = apply_scaler(X_train, scaler)
        X_val_scaled = apply_scaler(X_val, scaler)
        X_full_scaled = apply_scaler(X, scaler)

        if algo == "catboost":
            class_weight_train = compute_class_weights_list(y_train)
            class_weight_full = compute_class_weights_list(y)
            scale_pos_weight_train = None
            scale_pos_weight_full = None
        elif algo == "xgb":
            class_weight_train = None
            class_weight_full = None
            scale_pos_weight_train = compute_scale_pos_weight(y_train)
            scale_pos_weight_full = compute_scale_pos_weight(y)
        else:
            class_weight_train = compute_class_weights_dict(y_train)
            class_weight_full = compute_class_weights_dict(y)
            scale_pos_weight_train = None
            scale_pos_weight_full = None

        prepared[algo] = {
            "cfg": cfg,
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_full": X_full_scaled,
            "y_train": y_train,
            "y_val": y_val,
            "y_full": y,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "class_weight_train": class_weight_train,
            "class_weight_full": class_weight_full,
            "scale_pos_weight_train": scale_pos_weight_train,
            "scale_pos_weight_full": scale_pos_weight_full,
        }

    return prepared


def main():
    print("Loading training data...")
    df = load_dataset(DATASETS / "train.parquet")

    print("Preparing datasets for each algorithm...")
    prepared = prepare_datasets(df)

    def objective(trial: optuna.Trial) -> float:
        algo = trial.suggest_categorical("algorithm", list(ALGORITHMS.keys()))
        data = prepared[algo]

        params = suggest_params(trial, algo)
        model = build_model(
            algo,
            params,
            class_weight=data["class_weight_train"],
            scale_pos_weight=data["scale_pos_weight_train"],
        )

        if algo == "catboost":
            train_pool = Pool(data["X_train"], data["y_train"])
            val_pool = Pool(data["X_val"], data["y_val"])
            model.fit(train_pool, eval_set=val_pool)
            y_pred = model.predict(val_pool)
        elif algo == "lgbm":
            model.fit(
                data["X_train"],
                data["y_train"],
                eval_set=[(data["X_val"], data["y_val"])],
                eval_metric="binary_logloss",
            )
            y_pred = model.predict(data["X_val"])
        elif algo == "xgb":
            model.fit(
                data["X_train"],
                data["y_train"],
                eval_set=[(data["X_val"], data["y_val"])],
                verbose=False,
            )
            y_pred = model.predict(data["X_val"])
        else:
            model.fit(data["X_train"], data["y_train"])
            y_pred = model.predict(data["X_val"])

        return f1_score(data["y_val"], y_pred)

    print(f"Starting joint Optuna search across algorithms ({N_TRIALS} trials)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_algo = study.best_trial.params["algorithm"]
    best_params = extract_algo_params(study.best_trial.params, best_algo)
    print(f"Best algorithm: {best_algo}")
    print(f"Best F1: {study.best_value:.4f}")
    print(f"Best parameters: {best_params}")

    # Train the best algorithm on full data
    data = prepared[best_algo]
    final_model = build_model(
        best_algo,
        best_params,
        class_weight=data["class_weight_full"],
        scale_pos_weight=data["scale_pos_weight_full"],
    )

    if best_algo == "catboost":
        full_pool = Pool(data["X_full"], data["y_full"])
        final_model.fit(full_pool)
        val_pool = Pool(data["X_val"], data["y_val"])
        y_pred = final_model.predict(val_pool)
    elif best_algo == "lgbm":
        final_model.fit(
            data["X_full"],
            data["y_full"],
            eval_set=[(data["X_val"], data["y_val"])],
            eval_metric="binary_logloss",
        )
        y_pred = final_model.predict(data["X_val"])
    elif best_algo == "xgb":
        final_model.fit(
            data["X_full"],
            data["y_full"],
            eval_set=[(data["X_val"], data["y_val"])],
            verbose=False,
        )
        y_pred = final_model.predict(data["X_val"])
    else:
        final_model.fit(data["X_full"], data["y_full"])
        y_pred = final_model.predict(data["X_val"])

    print("\n=== Validation Performance (best algorithm/params) ===")
    metrics = compute_metrics(data["y_val"], y_pred)
    print_metrics(metrics)

    save_model(
        model_name=f"{MODEL_NAME}_{best_algo}",
        model=final_model,
        scaler=data["scaler"],
        feature_cols=data["feature_cols"],
        metrics=metrics,
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
        },
    )


if __name__ == "__main__":
    main()
