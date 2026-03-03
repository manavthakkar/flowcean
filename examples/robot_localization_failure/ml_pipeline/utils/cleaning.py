import shutil
from .paths import ARTIFACTS, DATASETS, MODELS

def clean_models():
    if MODELS.exists():
        shutil.rmtree(MODELS)
        print("Deleted artifacts/models/")
    MODELS.mkdir(parents=True, exist_ok=True)
    print("Recreated artifacts/models/")


def clean_dataset():
    DATASETS.mkdir(parents=True, exist_ok=True)
    targets = ["train.parquet", "eval.parquet", "train.csv", "eval.csv"]

    deleted = []
    for name in targets:
        path = DATASETS / name
        if path.exists():
            path.unlink()
            deleted.append(name)
            print(f"Deleted artifacts/datasets/{name}")

    if not deleted:
        print("No dataset files to delete in artifacts/datasets/")
    else:
        print("Finished cleaning dataset files in artifacts/datasets/")
