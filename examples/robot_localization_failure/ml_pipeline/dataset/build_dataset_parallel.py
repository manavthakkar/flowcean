import polars as pl
import flowcean.cli
import tempfile
import shutil
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from ml_pipeline.dataset.bag_processor import process_single_bag
from ml_pipeline.utils.paths import DATASETS
from ml_pipeline.dataset.helpers import get_topics


def safe_iter(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


# ============================================================
# PARALLEL WORKER FUNCTION (NO TQDM INSIDE WORKER)
# ============================================================
def worker_process_single_bag(args):
    """
    Worker process: runs in parallel.
    Computes ONE bag → writes ONE parquet file into temp_dir.
    No tqdm here (prevents flickering).
    """
    # Disable tqdm inside transforms
    os.environ["FLOWCEAN_PARALLEL"] = "1"

    bag_path, topics, msg_paths, pos_th, head_th, temp_dir, tag = args

    df = process_single_bag(
        bag_path=bag_path,
        topics=topics,
        message_paths=msg_paths,
        position_threshold=pos_th,
        heading_threshold=head_th,
    )

    temp_file = Path(temp_dir) / f"{tag}__{Path(bag_path).stem}.parquet"
    df.write_parquet(temp_file)
    return str(temp_file)


# ============================================================
# FINAL CONCAT + SORT (MAIN PROCESS)
# ============================================================
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


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    config = flowcean.cli.initialize()
    topics = get_topics()

    pos_th = float(config.localization.position_threshold)
    head_th = float(config.localization.heading_threshold)

    training_paths = safe_iter(config.rosbag.training_paths)
    eval_paths = safe_iter(config.rosbag.evaluation_paths)
    msg_paths = config.rosbag.message_paths

    # Create temp directory for parquet chunks
    temp_dir = tempfile.mkdtemp(prefix="flowcean_tmp_")
    print(f"🗂 Temp directory: {temp_dir}")

    try:
        # ============================================================
        # TRAINING SET (PARALLEL WITH MAIN-PROCESS tqdm)
        # ============================================================
        train_temp_files = []
        if training_paths:
            print(f"\n📦 Processing {len(training_paths)} training bag(s) in parallel...\n")

            tasks = [
                (
                    bag,
                    topics,
                    msg_paths,
                    pos_th,
                    head_th,
                    temp_dir,
                    "train",
                )
                for bag in training_paths
            ]

            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(worker_process_single_bag, t) for t in tasks]

                # Main-process tqdm bar
                for f in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Training bags processed",
                ):
                    train_temp_files.append(Path(f.result()))

            print("\n🧮 Finalizing training dataset...")
            finalize_dataset(
                train_temp_files,
                output_parquet=DATASETS / "train.parquet",
                output_csv=DATASETS / "train.csv",
            )
            print("✔ Training dataset saved.\n")
        else:
            print("⚠️ No training bags provided.")

        # ============================================================
        # EVALUATION SET (PARALLEL WITH MAIN-PROCESS tqdm)
        # ============================================================
        eval_temp_files = []
        if eval_paths:
            print(f"\n📦 Processing {len(eval_paths)} evaluation bag(s) in parallel...\n")

            tasks = [
                (
                    bag,
                    topics,
                    msg_paths,
                    pos_th,
                    head_th,
                    temp_dir,
                    "eval",
                )
                for bag in eval_paths
            ]

            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(worker_process_single_bag, t) for t in tasks]

                for f in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Evaluation bags processed",
                ):
                    eval_temp_files.append(Path(f.result()))

            print("\n🧮 Finalizing evaluation dataset...")
            finalize_dataset(
                eval_temp_files,
                output_parquet=DATASETS / "eval.parquet",
                output_csv=DATASETS / "eval.csv",
            )
            print("✔ Evaluation dataset saved.\n")
        else:
            print("⚠️ No evaluation bags provided.")

        if not training_paths and not eval_paths:
            print("\n❌ ERROR: No training or evaluation paths found.")
            return

    finally:
        # ============================================================
        # CLEANUP TEMP FILES
        # ============================================================
        print("\n🧹 Cleaning up temporary files...")
        try:
            shutil.rmtree(temp_dir)
            print(f"✔ Temp directory deleted: {temp_dir}")
        except Exception as e:
            print(f"⚠️ Warning: Could not delete temp directory: {e}")

    print("\n🎉 Dataset build complete.")


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()

    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
