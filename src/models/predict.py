"""Generación de predicciones con el modelo ya entrenado y persistido.

Usado tanto por `main.py predict` (batch, CLI) como por `src/api/routes.py`
(online, un cliente a la vez). Traduce la probabilidad de compra en una
banda de propensión (alta/media/baja) siguiendo la estrategia de campaña
propuesta en el notebook exploratorio.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.exceptions import PredictionError
from src.models.save_model import load_pipeline

logger = logging.getLogger(__name__)


def assign_propensity_band(probability: float, high_threshold: float, medium_threshold: float) -> str:
    if probability >= high_threshold:
        return "alta"
    if probability >= medium_threshold:
        return "media"
    return "baja"


def predict_batch(
    pipeline: Any,
    df: pd.DataFrame,
    feature_columns: list[str],
    high_threshold: float,
    medium_threshold: float,
) -> pd.DataFrame:
    """Genera predicción binaria, probabilidad y banda de propensión para cada fila de `df`."""
    missing_cols = set(feature_columns) - set(df.columns)
    if missing_cols:
        raise PredictionError(f"Faltan columnas requeridas para predecir: {sorted(missing_cols)}")

    try:
        X = df[feature_columns]
        predictions = pipeline.predict(X)
        probabilities = pipeline.predict_proba(X)[:, 1]
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(f"Fallo al generar predicciones: {exc}") from exc

    result = df.copy()
    result["prediction"] = predictions
    result["probability"] = probabilities
    result["propensity_band"] = [
        assign_propensity_band(p, high_threshold, medium_threshold) for p in probabilities
    ]
    logger.info("Predicciones generadas para %d registros.", len(result))
    return result


def predict_single(
    pipeline: Any,
    record: dict[str, Any],
    feature_columns: list[str],
    high_threshold: float,
    medium_threshold: float,
) -> dict[str, Any]:
    """Versión de `predict_batch` para un único registro (usada por la API)."""
    df = pd.DataFrame([record])
    result_df = predict_batch(pipeline, df, feature_columns, high_threshold, medium_threshold)
    row = result_df.iloc[0]
    return {
        "prediction": int(row["prediction"]),
        "probability": float(row["probability"]),
        "propensity_band": str(row["propensity_band"]),
    }


def load_model_for_inference(model_path: str) -> Any:
    return load_pipeline(model_path)
