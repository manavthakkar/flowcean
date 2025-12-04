#!/usr/bin/env python3
import flowcean.cli
from pathlib import Path
import polars as pl

from ml_pipeline.utils.paths import DATASETS


PROCESSED_DIR = DATASETS / "processed"


def load_bag_table(bag_path: str) -> pl.DataFrame:
    bag_name = Path(bag_path).stem
    parquet_path = PROCESSED_DIR / f"{bag_name}.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Processed parquet not found for:\n{bag_path}\n"
            f"Expected here:\n{parquet_path}\n"
            f"Run process_bags.py first!"
        )

    print(f"→ Loading {parquet_path}")
    return pl.read_parquet(parquet_path)


def build_dataset(bag_paths, parquet_out, csv_out):
    if not bag_paths:
        print(f"⚠ No bags provided for {parquet_out}")
        return

    tables = [load_bag_table(path) for path in bag_paths]
    df = pl.concat(tables, how="vertical").sort("time")

    df.write_parquet(parquet_out)
    df.write_csv(csv_out)

    print(f"✔ Saved: {parquet_out}")
    print(f"✔ Saved: {csv_out}")


def main():
    config = flowcean.cli.initialize()

    train_paths = config.rosbag.training_paths
    eval_paths = config.rosbag.evaluation_paths

    print("\n=== Building TRAIN dataset from rosbag.training_paths ===")
    build_dataset(
        train_paths,
        DATASETS / "train.parquet",
        DATASETS / "train.csv",
    )

    print("\n=== Building EVAL dataset from rosbag.evaluation_paths ===")
    build_dataset(
        eval_paths,
        DATASETS / "eval.parquet",
        DATASETS / "eval.csv",
    )

    print("\n🎉 Train/Eval dataset successfully built!\n")


if __name__ == "__main__":
    main()
