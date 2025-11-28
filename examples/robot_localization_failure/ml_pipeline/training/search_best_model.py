import numpy as np
import optuna
from optuna import TrialPruned

from sklearn.metrics import f1_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from ml_pipeline.utils.paths import DATASETS
from ml_pipeline.utils.common import (
    load_dataset,
    prepare_features,
    add_temporal_features,
    compute_metrics,
    print_metrics,
    save_model,
)


# Basic search configuration
MODEL_NAME = "feature_hparam_search"
N_TRIALS = 30
RANDOM_STATE = 42


def create_model(model_type: str, trial: optuna.Trial, y_train):
    """
    Build a model for the requested type using trial-suggested params.
    """
    from sklearn.ensemble import RandomForestClassifier

    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 200, 800, step=100),
            max_depth=trial.suggest_int("rf_max_depth", 4, 20),
            min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 5),
            max_features=trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    if model_type == "xgb" and HAS_XGB:
        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            n_estimators=trial.suggest_int("xgb_n_estimators", 200, 800, step=100),
            max_depth=trial.suggest_int("xgb_max_depth", 4, 12),
            learning_rate=trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_float("xgb_min_child_weight", 1.0, 10.0),
            gamma=trial.suggest_float("xgb_gamma", 0.0, 5.0),
            reg_lambda=trial.suggest_float("xgb_reg_lambda", 0.0, 5.0),
            reg_alpha=trial.suggest_float("xgb_reg_alpha", 0.0, 5.0),
            scale_pos_weight=trial.suggest_float("xgb_scale_pos_weight", 1.0, max(1.0, neg / max(1, pos))),
        )

    raise ValueError(f"Unsupported or unavailable model_type: {model_type}")


def main():
    train_path = DATASETS / "train.parquet"
    eval_path = DATASETS / "eval.parquet"

    print(f"Loading training data from {train_path}")
    base_train_df = load_dataset(train_path)
    print(f"Loading eval data from {eval_path}")
    base_eval_df = load_dataset(eval_path)

    # Pre-compute temporal variants to avoid recomputing inside the objective
    temporal_train_df = add_temporal_features(base_train_df)
    temporal_eval_df = add_temporal_features(base_eval_df)

    # Cache prepared numpy arrays keyed by feature toggles
    prepared_cache: dict[tuple[bool, bool, bool, bool], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}

    def get_prepared(
        use_temporal: bool,
        use_scanmap: bool,
        use_particle: bool,
        use_amcl: bool,
    ):
        key = (use_temporal, use_scanmap, use_particle, use_amcl)
        if key in prepared_cache:
            return prepared_cache[key]

        train_df = temporal_train_df if use_temporal else base_train_df
        eval_df = temporal_eval_df if use_temporal else base_eval_df

        try:
            X_train, y_train, feature_cols = prepare_features(
                train_df,
                use_scanmap_features=use_scanmap,
                use_particle_features=use_particle,
                use_amcl_pose=use_amcl,
            )
            X_eval, y_eval, _ = prepare_features(
                eval_df,
                use_scanmap_features=use_scanmap,
                use_particle_features=use_particle,
                use_amcl_pose=use_amcl,
            )
        except ValueError as e:
            # No usable columns for this toggle combo
            raise TrialPruned(f"Feature prep failed: {e}")

        if X_train.shape[1] == 0 or X_eval.shape[1] == 0:
            raise TrialPruned("No usable features for this toggle combination.")

        prepared_cache[key] = (X_train, y_train, X_eval, y_eval, feature_cols)
        return prepared_cache[key]

    def objective(trial: optuna.Trial) -> float:
        # Feature toggles
        use_temporal = trial.suggest_categorical("use_temporal_features", [True, False])
        use_scanmap = trial.suggest_categorical("use_scanmap_features", [True, False])
        use_particle = trial.suggest_categorical("use_particle_features", [True, False])
        use_amcl = trial.suggest_categorical("use_amcl_pose", [True, False])

        X_train, y_train, X_eval, y_eval, feature_cols = get_prepared(
            use_temporal,
            use_scanmap,
            use_particle,
            use_amcl,
        )

        model_types = ["rf"]
        if HAS_XGB:
            model_types.append("xgb")

        model_type = trial.suggest_categorical("model_type", model_types)
        model = create_model(model_type, trial, y_train)

        model.fit(X_train, y_train)
        preds = model.predict(X_eval)

        # Use F1 on the eval set as the objective
        score = f1_score(y_eval, preds)
        return score

    print(f"\n🚀 Starting Optuna study with {N_TRIALS} trials, optimizing F1 on eval.parquet")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n==============================")
    print("🎯 Optuna Best Trial")
    print("==============================")
    print(f"Best F1 on eval: {study.best_value:.4f}")
    print("Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best_params = dict(study.best_params)
    best_model_type = best_params["model_type"]
    use_temporal = best_params["use_temporal_features"]
    use_scanmap = best_params["use_scanmap_features"]
    use_particle = best_params["use_particle_features"]
    use_amcl = best_params["use_amcl_pose"]

    X_train, y_train, X_eval, y_eval, feature_cols = get_prepared(
        use_temporal,
        use_scanmap,
        use_particle,
        use_amcl,
    )

    # Rebuild best model using FixedTrial to reuse param logic
    fixed_trial = optuna.trial.FixedTrial(best_params)
    best_model = create_model(best_model_type, fixed_trial, y_train)

    print(f"\nRe-training best {best_model_type} with discovered feature set on full train set...")
    best_model.fit(X_train, y_train)

    eval_preds = best_model.predict(X_eval)
    eval_metrics = compute_metrics(y_eval, eval_preds)

    print("\n=== Eval metrics for best model ===")
    print_metrics(eval_metrics)

    extra_metadata = {
        "search_trials": N_TRIALS,
        "best_params": best_params,
        "best_eval_f1": float(study.best_value),
        "model_type": best_model_type,
        "use_temporal_features": use_temporal,
        "use_scanmap_features": use_scanmap,
        "use_particle_features": use_particle,
        "use_amcl_pose": use_amcl,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
    }

    save_model(
        model_name=f"{MODEL_NAME}_{best_model_type}",
        model=best_model,
        scaler=None,
        feature_cols=feature_cols,
        metrics=eval_metrics,
        extra_metadata=extra_metadata,
    )


if __name__ == "__main__":
    main()
