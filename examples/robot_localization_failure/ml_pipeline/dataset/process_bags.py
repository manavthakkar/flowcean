#!/usr/bin/env python3
import flowcean.cli
from pathlib import Path
import polars as pl

from ml_pipeline.dataset.bag_processor import process_single_bag
from ml_pipeline.dataset.helpers import get_topics
from ml_pipeline.utils.paths import DATASETS


# Output folder: artifacts/datasets/processed
PROCESSED_DIR = DATASETS / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_output_paths(bag_path: str):
    bag_name = Path(bag_path).stem
    return (
        PROCESSED_DIR / f"{bag_name}.parquet",
        PROCESSED_DIR / f"{bag_name}.csv",
    )


def process_bag(bag_path: str, config, topics):
    parquet_path, csv_path = get_output_paths(bag_path)

    # Skip if exists
    if parquet_path.exists():
        print(f"✔ Already processed: {parquet_path}")
        return

    print(f"\n Processing bag: {bag_path}")

    df = process_single_bag(
        bag_path=bag_path,
        topics=topics,
        message_paths=config.rosbag.message_paths,
        position_threshold=float(config.localization.position_threshold),
        heading_threshold=float(config.localization.heading_threshold),
    )

    df.write_parquet(parquet_path)
    df.write_csv(csv_path)

    print(f"✔ Saved: {parquet_path}")
    print(f"✔ Saved: {csv_path}")


def main():
    config = flowcean.cli.initialize()
    topics = get_topics()

    # bags from config.yaml
    all_bags = config.rosbag.bags
    failed_bags = []

    print("\n=== Processing ALL Bags from config.yaml → rosbag.bags ===")
    for bag_path in all_bags:
        try:
            process_bag(bag_path, config, topics)
        except Exception as exc:
            failed_bags.append((bag_path, exc))
            print(f"✖ Failed to process {bag_path}: {exc}")

    if failed_bags:
        print("\n Completed with errors:")
        for bag_path, exc in failed_bags:
            print(f"- {bag_path}: {exc}")
    else:
        print("\n Finished processing all bags!\n")


if __name__ == "__main__":
    main()
