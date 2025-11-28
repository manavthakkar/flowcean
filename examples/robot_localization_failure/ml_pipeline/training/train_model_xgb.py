# ml_pipeline/training/train_model_xgb.py

import xgboost as xgb
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

MODEL_NAME = "xgboost"          #xgboost_temporal
USE_TEMPORAL_FEATURES = True
USE_SCANMAP_FEATURES = True
USE_PARTICLE_FEATURES = True
USE_AMCL_POSE = True


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

    # ============================================================
    # 4) Train model
    # ============================================================
    print(f"\nTraining {MODEL_NAME} with a small param sweep...")

    # simple imbalance weight
    pos = sum(y)
    neg = len(y) - pos
    base_scale_pos_weight = neg / max(1, pos)

    candidate_params = [
        {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
    ]

    best = None
    for i, params in enumerate(candidate_params):
        print(f"\n[Trial {i+1}/{len(candidate_params)}] params={params}")
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            reg_lambda=1.0,
            reg_alpha=0.0,
            scale_pos_weight=base_scale_pos_weight,
            random_state=42,
            **params,
        )

        model.fit(X_train_scaled, y_train, verbose=False)
        y_pred = model.predict(X_val_scaled)
        metrics = compute_metrics(y_val, y_pred)
        print_metrics(metrics)

        f1 = metrics["f1"]
        if best is None or f1 > best["f1"]:
            best = {"f1": f1, "metrics": metrics, "model": model, "params": params}

    assert best is not None, "No model was trained."
    model = best["model"]
    metrics = best["metrics"]

    print("\n=== Selected best XGBoost model ===")
    print(f"F1: {best['f1']:.4f} | params: {best['params']}")

    # ============================================================
    # 6) Save model package
    # ============================================================
    save_model(
        model_name=MODEL_NAME,
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        metrics=metrics,
        extra_metadata={
            "temporal_features": USE_TEMPORAL_FEATURES,
            "model_type": MODEL_NAME,
            "notes": "temporal + scan-map features; small manual sweep",
            "use_scanmap_features": USE_SCANMAP_FEATURES,
            "use_particle_features": USE_PARTICLE_FEATURES,
            "use_amcl_pose": USE_AMCL_POSE,
            "best_params": best["params"],
        }
    )


if __name__ == "__main__":
    main()
