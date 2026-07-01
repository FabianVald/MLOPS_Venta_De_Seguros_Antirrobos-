"""Integración con MLflow Tracking y Model Registry.

Centraliza el `set_tracking_uri`/`set_experiment` y expone un context manager
(`mlflow_run`) para que `train.py` registre, en cada corrida: parámetros,
métricas, matriz de confusión, curva ROC, feature importance y el propio
modelo entrenado, sin repetir boilerplate de MLflow en cada script.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)


def _normalize_tracking_uri(tracking_uri: str) -> str:
    """Convierte rutas locales (relativas o absolutas) a un `file://` URI válido.

    mlflow interpreta cualquier string sin esquema como ruta de archivo, pero
    en Windows una ruta absoluta como 'C:\\...\\mlruns' es mal interpretada:
    el 'C:' se lee como esquema de URI. Resolver siempre a `Path(...).as_uri()`
    evita ese problema y funciona igual en Linux/Mac.
    """
    if "://" in tracking_uri:
        return tracking_uri
    return Path(tracking_uri).resolve().as_uri()


def configure_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Configura el tracking URI y el experimento activo de MLflow."""
    resolved_uri = _normalize_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow configurado -> tracking_uri=%s, experiment=%s", resolved_uri, experiment_name)


@contextmanager
def mlflow_run(run_name: str, tags: dict[str, str] | None = None) -> Iterator[Any]:
    """Context manager que abre y cierra un run de MLflow de forma segura."""
    run = mlflow.start_run(run_name=run_name, tags=tags)
    logger.info("MLflow run iniciado: %s (id=%s)", run_name, run.info.run_id)
    try:
        yield run
    finally:
        mlflow.end_run()
        logger.info("MLflow run finalizado: %s", run_name)


def log_params(params: dict[str, Any]) -> None:
    flat_params = {str(k): v for k, v in params.items() if v is not None}
    mlflow.log_params(flat_params)


def log_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics(metrics)


def log_figure_artifact(figure_path: str) -> None:
    mlflow.log_artifact(figure_path, artifact_path="figures")


def log_sklearn_model(pipeline: Any, artifact_path: str = "model", registered_model_name: str | None = None) -> None:
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
    )
    logger.info("Modelo registrado en MLflow bajo artifact_path='%s'", artifact_path)
