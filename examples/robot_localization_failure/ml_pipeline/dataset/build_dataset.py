import polars as pl
import flowcean.cli
import tempfile
import shutil
from pathlib import Path

from ml_pipeline.dataset.bag_processor import process_single_bag
from ml_pipeline.utils.paths import DATASETS
from ml_pipeline.dataset.helpers import get_topics


def safe_iter(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def process_and_write_temp(bag_path, topics, msg_paths, pos_th, head_th, temp_dir, tag):
    df = process_single_bag(
        bag_path=bag_path,
        topics=topics,
        message_paths=msg_paths,
        position_threshold=pos_th,
        heading_threshold=head_th,
    )

    temp_file = Path(temp_dir) / f"{tag}__{Path(bag_path).stem}.parquet"
    df.write_parquet(temp_file)
    return temp_file


def finalize_dataset(temp_files, output_parquet, output_csv):
    if not temp_files:
        return

    lazy_frames = [pl.scan_parquet(str(f)) for f in temp_files]
    final_df = (
        pl.concat(lazy_frames, how="vertical")
        .sort("time", maintain_order=True)
        .collect()
    )

    final_df.write_parquet(output_parquet)
    final_df.write_csv(output_csv)


def main():
    config = flowcean.cli.initialize()
    topics = get_topics()

    pos_th = float(config.localization.position_threshold)
    head_th = float(config.localization.heading_threshold)

    training_paths = safe_iter(config.rosbag.training_paths)
    eval_paths = safe_iter(config.rosbag.evaluation_paths)
    msg_paths = config.rosbag.message_paths

    # Create a temp directory for all parquet chunks
    temp_dir = tempfile.mkdtemp(prefix="flowcean_tmp_")
    temp_dir_path = Path(temp_dir)

    print(f"🗂 Temp directory created: {temp_dir}")

    # Track all temp files for cleanup
    temp_files_to_delete = []

    try:
        # ============================
        # TRAINING
        # ============================
        train_temp_files = []
        if training_paths:
            print(f"📦 Processing {len(training_paths)} training bag(s)...")

            for bag in training_paths:
                print(f" → Training bag: {bag}")
                tfile = process_and_write_temp(
                    bag, topics, msg_paths, pos_th, head_th, temp_dir, "train"
                )
                train_temp_files.append(tfile)
                temp_files_to_delete.append(tfile)

            print("🧮 Finalizing training dataset...")
            finalize_dataset(
                train_temp_files,
                output_parquet=DATASETS / "train.parquet",
                output_csv=DATASETS / "train.csv",
            )
            print("✔ Training dataset saved.")
        else:
            print("⚠️ No training bags provided.")

        # ============================
        # EVALUATION
        # ============================
        eval_temp_files = []
        if eval_paths:
            print(f"📦 Processing {len(eval_paths)} evaluation bag(s)...")

            for bag in eval_paths:
                print(f" → Evaluation bag: {bag}")
                tfile = process_and_write_temp(
                    bag, topics, msg_paths, pos_th, head_th, temp_dir, "eval"
                )
                eval_temp_files.append(tfile)
                temp_files_to_delete.append(tfile)

            print("🧮 Finalizing evaluation dataset...")
            finalize_dataset(
                eval_temp_files,
                output_parquet=DATASETS / "eval.parquet",
                output_csv=DATASETS / "eval.csv",
            )
            print("✔ Evaluation dataset saved.")
        else:
            print("⚠️ No evaluation bags provided.")

        if not training_paths and not eval_paths:
            print("\n❌ ERROR: No training or evaluation paths found.")
            return

    finally:
        # ============================
        # CLEANUP
        # ============================
        print("\n🧹 Cleaning up temporary files...")

        try:
            shutil.rmtree(temp_dir)
            print(f"✔ Temp directory deleted: {temp_dir}")
        except Exception as e:
            print(f"⚠️ Warning: Could not delete temp directory: {e}")

    print("\n🎉 Dataset build complete.")


if __name__ == "__main__":
    main()
