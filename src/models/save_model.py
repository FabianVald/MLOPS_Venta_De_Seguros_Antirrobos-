"""Persistencia del modelo (pipeline completo) en disco, fuera de MLflow.

Además del registro en MLflow (auditable/versionado), la API de serving
carga el modelo directamente desde `models/model.pkl` para no depender de
que el MLflow tracking server esté disponible en producción.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib

from src.exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)


def save_pipeline(pipeline: Any, path: str) -> None:
    """Serializa el pipeline entrenado (preprocesamiento + modelo) con joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info("Pipeline guardado en %s", output_path)


def load_pipeline(path: str) -> Any:
    """Carga un pipeline previamente serializado. Lanza ModelNotFoundError si no existe."""
    input_path = Path(path)
    if not input_path.exists():
        raise ModelNotFoundError(
            f"No existe un modelo entrenado en '{path}'. Ejecuta el pipeline de entrenamiento primero."
        )
    logger.info("Cargando pipeline desde %s", input_path)
    return joblib.load(input_path)


def save_metadata(metadata: dict[str, Any], path: str) -> None:
    """Guarda metadatos del modelo (algoritmo, métricas, features, fecha) como JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Metadata del modelo guardada en %s", output_path)


def load_metadata(path: str) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise ModelNotFoundError(f"No existe metadata del modelo en '{path}'.")
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
