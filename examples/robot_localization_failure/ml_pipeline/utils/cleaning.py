import shutil
from .paths import ARTIFACTS, MODELS

def clean_artifacts():
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
        print("✔ Deleted artifacts/")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print("✔ Recreated artifacts/")


def clean_models():
    if MODELS.exists():
        shutil.rmtree(MODELS)
        print("✔ Deleted artifacts/models/")
    MODELS.mkdir(parents=True, exist_ok=True)
    print("✔ Recreated artifacts/models/")
