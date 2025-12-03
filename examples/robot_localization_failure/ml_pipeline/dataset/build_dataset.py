import polars as pl
import flowcean.cli
import tempfile
import os
from pathlib import Path

from ml_pipeline.dataset.bag_processor import process_single_bag
from ml_pipeline.utils.paths import DATASETS
from ml_pipeline.dataset.helpers import get_topics


def safe_iter(x):
    """
    Convert None → empty list.
    Convert single string → [string].
    Return list unchanged.
    """
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def process_and_write_temp(bag_path, topics, msg_paths, pos_th, head_th, temp_dir, tag):
    """
    Process a single bag, write to a temporary parquet file.
    Returns the path to the parquet.
    """
    print(f" → Processing bag: {bag_path}")

    df = process_single_bag(
        bag_path=bag_path,
        topics=topics,
        message_paths=msg_paths,
        position_threshold=pos_th,
        heading_threshold=head_th,
    )

    temp_path = Path(temp_dir) / f"{tag}__{Path(bag_path).stem}.parquet"
    df.write_parquet(temp_path)

    return temp_path


def finalize_dataset(temp_files, output_parquet, output_csv):
    """
    Use Polars LazyFrame streaming to concat, sort, and write final outputs.
    Guarantees identical ordering to original implementation.
    """
    if len(temp_files) == 0:
        return

    # Lazy loading of each parquet file (streaming)
    lazy_frames = [pl.scan_parquet(str(f)) for f in temp_files]

    final_df = (
        pl.concat(lazy_frames, how="vertical")
        .sort("time")   # identical to original behavior
        .collect()
    )

    final_df.write_parquet(output_parquet)
    final_df.write_csv(output_csv)


def main():
    # Load config
    config = flowcean.cli.initialize()
    topics = get_topics()

    pos_th = float(config.localization.position_threshold)
    head_th = float(config.localization.heading_threshold)

    training_paths = safe_iter(config.rosbag.training_paths)
    eval_paths = safe_iter(config.rosbag.evaluation_paths)

    msg_paths = config.rosbag.message_paths

    # Temporary directory for streaming parquet files
    temp_dir = tempfile.mkdtemp(prefix="flowcean_tmp_")
    print(f"🗂 Temporary directory: {temp_dir}")

    # ============================================================
    # TRAINING SET
    # ============================================================
    train_temp_files = []
    if training_paths:
        print(f"📦 Processing {len(training_paths)} training bag(s)...")

        for bag in training_paths:
            temp_file = process_and_write_temp(
                bag, topics, msg_paths, pos_th, head_th, temp_dir, tag="train"
            )
            train_temp_files.append(temp_file)

        print("🧮 Finalizing training dataset...")
        finalize_dataset(
            temp_files=train_temp_files,
            output_parquet=DATASETS / "train.parquet",
            output_csv=DATASETS / "train.csv",
        )

        print("✔ Saved training dataset")
    else:
        print("⚠️ No training bags found — skipping training dataset creation.")

    # ============================================================
    # EVALUATION SET
    # ============================================================
    eval_temp_files = []
    if eval_paths:
        print(f"📦 Processing {len(eval_paths)} evaluation bag(s)...")

        for bag in eval_paths:
            temp_file = process_and_write_temp(
                bag, topics, msg_paths, pos_th, head_th, temp_dir, tag="eval"
            )
            eval_temp_files.append(temp_file)

        print("🧮 Finalizing evaluation dataset...")
        finalize_dataset(
            temp_files=eval_temp_files,
            output_parquet=DATASETS / "eval.parquet",
            output_csv=DATASETS / "eval.csv",
        )

        print("✔ Saved evaluation dataset")
    else:
        print("⚠️ No evaluation bags found — skipping evaluation dataset creation.")

    # ============================================================
    # FINAL SANITY CHECK
    # ============================================================
    if not training_paths and not eval_paths:
        print("\n❌ ERROR: Both training_paths and evaluation_paths are empty.")
        print("Nothing to process. Check your config.yaml.")
        return

    print(f"\n🧹 Temporary parquet files stored at: {temp_dir}")
    print("Delete this directory manually after verification.")


if __name__ == "__main__":
    main()
