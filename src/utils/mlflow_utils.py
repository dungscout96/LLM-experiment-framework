"""MLflow utilities for experiment tracking."""

import logging
from pathlib import Path
from typing import Any

import mlflow
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def setup_mlflow(cfg: DictConfig) -> str:
    """Set up MLflow tracking.

    Args:
        cfg: Configuration with mlflow settings.

    Returns:
        Experiment ID.
    """
    tracking_uri = cfg.mlflow.tracking_uri
    experiment_name = cfg.mlflow.experiment_name

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow tracking URI: {tracking_uri}")
    logger.info(f"Experiment: {experiment_name} (ID: {experiment_id})")

    return experiment_id


def log_config(cfg: DictConfig) -> None:
    """Log configuration to MLflow.

    Args:
        cfg: Hydra configuration.
    """
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    def flatten_dict(d: dict, parent_key: str = "") -> dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_config = flatten_dict(config_dict)

    for key, value in flat_config.items():
        if isinstance(value, (str, int, float, bool)):
            mlflow.log_param(key, value)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to MLflow.

    Args:
        metrics: Dictionary of metric names and values.
        step: Optional step number.
    """
    mlflow.log_metrics(metrics, step=step)


def log_model_artifact(
    model_path: str | Path,
    artifact_path: str = "model",
) -> None:
    """Log model artifacts to MLflow.

    Args:
        model_path: Path to saved model.
        artifact_path: MLflow artifact path.
    """
    mlflow.log_artifacts(str(model_path), artifact_path=artifact_path)
    logger.info(f"Model artifacts logged to: {artifact_path}")


def load_best_model(
    experiment_name: str,
    metric_name: str = "eval_loss",
    maximize: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Find and return the best model from MLflow experiments.

    Args:
        experiment_name: Name of the MLflow experiment.
        metric_name: Metric to use for comparison.
        maximize: Whether to maximize the metric.

    Returns:
        Tuple of (run_id, model_artifact_uri).
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} {'DESC' if maximize else 'ASC'}"],
    )

    if runs.empty:
        raise ValueError(f"No runs found in experiment: {experiment_name}")

    best_run = runs.iloc[0]
    run_id = best_run["run_id"]

    artifact_uri = f"runs:/{run_id}/model"

    return run_id, {
        "run_id": run_id,
        "artifact_uri": artifact_uri,
        f"{metric_name}": best_run[f"metrics.{metric_name}"],
    }


def compare_runs(
    experiment_name: str,
    metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compare runs in an experiment.

    Args:
        experiment_name: Name of the experiment.
        metrics: List of metrics to compare.

    Returns:
        List of run summaries.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    summaries = []
    for _, run in runs.iterrows():
        summary = {
            "run_id": run["run_id"],
            "run_name": run.get("tags.mlflow.runName", ""),
            "status": run["status"],
            "start_time": run["start_time"],
        }

        if metrics:
            for metric in metrics:
                col = f"metrics.{metric}"
                if col in run:
                    summary[metric] = run[col]

        summaries.append(summary)

    return summaries
