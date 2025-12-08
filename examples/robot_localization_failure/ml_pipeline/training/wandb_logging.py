# ml_pipeline/training/wandb_logging.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Any

try:
    import wandb
except Exception:  # wandb not installed / disabled
    wandb = None  # type: ignore


def init_wandb_run(
    config_obj: Any,
    run_name: str,
    config_dict: Dict[str, Any],
) -> Optional["wandb.sdk.wandb_run.Run"]:
    """
    Initialize a Weights & Biases run if wandb is available.
    Returns the run object or None.
    """
    if wandb is None:
        print("⚠️ Weights & Biases not installed or disabled, skipping W&B logging.")
        return None

    # Optional project/entity from config (if present)
    project = getattr(getattr(config_obj, "wandb", None), "project", "robot_localization_monitoring")
    entity = getattr(getattr(config_obj, "wandb", None), "entity", None)

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config_dict,
    )
    return run


def log_optuna_summary(
    run,
    best_algo: str,
    best_params: Dict[str, Any],
    best_f1: float,
    n_trials: int,
) -> None:
    if run is None:
        return

    run.config.update({"best_algorithm": best_algo}, allow_val_change=True)
    run.config.update({"best_params": best_params}, allow_val_change=True)
    run.log({
        "optuna/best_f1": best_f1,
        "optuna/n_trials": n_trials,
    })


def log_eval_metrics_to_wandb(
    run,
    metrics_t05: Dict[str, float],
    best_thr: float,
    best_metrics: Dict[str, float],
) -> None:
    if run is None:
        return

    run.log({
        "eval/f1_t05": metrics_t05["f1"],
        "eval/precision_t05": metrics_t05["precision"],
        "eval/recall_t05": metrics_t05["recall"],
        "eval/best_threshold_f1": best_thr,
        "eval/f1_best": best_metrics["f1"],
        "eval/precision_best": best_metrics["precision"],
        "eval/recall_best": best_metrics["recall"],
    })


def log_plots_to_wandb(
    run,
    plot_paths: Dict[str, Path],
) -> None:
    if run is None or wandb is None:
        return

    for name, path in plot_paths.items():
        run.log({f"plots/{name}": wandb.Image(str(path))})


def log_model_artifact(
    run,
    model_dir: Path,
    artifact_name: str,
) -> None:
    if run is None or wandb is None:
        return

    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
    )
    artifact.add_dir(model_dir)
    run.log_artifact(artifact)


def finish_wandb_run(run) -> None:
    if run is None:
        return
    run.finish()
