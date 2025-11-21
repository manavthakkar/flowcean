import polars as pl
from pathlib import Path
from ml_pipeline.utils.paths import DATASETS


def add_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add temporal (rolling/differential) features to an already-clean ML dataset.
    Does NOT modify the original dataset pipeline.
    """
    df = df.sort("time")

    # Columns that must NOT be used to create temporal features
    exclude = {
        "time",
        "is_delocalized",
        "gt_x", "gt_y", "gt_qx", "gt_qy", "gt_qz", "gt_qw",
        "gt_yaw",
        "position_error", "heading_error_raw", "heading_error",
        "combined_error",
    }

    numeric_types = (pl.Float32, pl.Float64, pl.Int64, pl.Int32)
    base_features = [
        c for c, dtype in df.schema.items()
        if c not in exclude and isinstance(dtype, numeric_types)
    ]

    print(f"[temporal] Base numeric features ({len(base_features)}):")
    print(base_features)

    new_cols = []
    for col in base_features:
        new_cols.extend([
            (pl.col(col) - pl.col(col).shift(1)).alias(f"{col}_diff1"),
            pl.col(col).rolling_mean(window_size=5, min_periods=1).alias(f"{col}_mean5"),
            pl.col(col).rolling_std(window_size=5, min_periods=1).alias(f"{col}_std5"),
        ])

    df = df.with_columns(new_cols)
    print(f"[temporal] Added {len(new_cols)} new temporal features.")
    return df


def main():
    print("Loading base dataset...")
    df = pl.read_parquet(DATASETS / "train.parquet")

    print("Applying temporal feature engineering...")
    df_t = add_temporal_features(df)

    out_path = DATASETS / "train_temporal.parquet"
    df_t.write_parquet(out_path)
    print(f"[temporal] Saved → {out_path}")

    # OPTIONAL: Do the same for eval
    if (DATASETS / "eval.parquet").exists():
        df_eval = pl.read_parquet(DATASETS / "eval.parquet")
        df_eval_t = add_temporal_features(df_eval)
        out_eval_path = DATASETS / "eval_temporal.parquet"
        df_eval_t.write_parquet(out_eval_path)
        print(f"[temporal] Saved → {out_eval_path}")


if __name__ == "__main__":
    main()
