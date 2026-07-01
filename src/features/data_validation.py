"""Validación de esquema de datos (crudos y procesados).

Sustituye a un framework externo (Great Expectations / Pandera) por un
validador ligero y explícito: para este dataset de 12 columnas no se
justifica una dependencia adicional, pero la interfaz (`validate_raw_schema`,
`validate_processed_schema`) es la que se reemplazaría si el proyecto
creciera y se quisiera migrar a esas herramientas.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.exceptions import DataValidationError

logger = logging.getLogger(__name__)

# Rango de negocio razonable para columnas numéricas críticas. Se usa solo
# para detectar corrupción de datos evidente (p. ej. edades negativas),
# no para descartar outliers legítimos (eso lo maneja el capping en preprocessing).
NUMERIC_BUSINESS_RANGES = {
    "EDAD": (0, 120),
    "ANTIGUEDAD_MES": (0, 1000),
    "Mto_TC": (0, None),
    "SUELDO_ESTIMADO": (0, None),
}


def validate_raw_schema(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
) -> None:
    """Valida que el dataset crudo tenga las columnas esperadas y tipos coherentes.

    Lanza DataValidationError con un mensaje descriptivo si algo no cumple.
    No modifica el DataFrame.
    """
    logger.info("Validando esquema de datos crudos (%d filas, %d columnas)", *df.shape)

    if df.empty:
        raise DataValidationError("El dataset crudo está vacío.")

    expected_cols = set(numeric_cols) | set(categorical_cols) | {target_col}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise DataValidationError(
            f"Faltan columnas obligatorias en el dataset crudo: {sorted(missing_cols)}"
        )

    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(
                f"La columna numérica '{col}' no tiene un dtype numérico (dtype actual: {df[col].dtype})."
            )

    for col, (low, high) in NUMERIC_BUSINESS_RANGES.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if low is not None and (series < low).any():
            raise DataValidationError(
                f"La columna '{col}' contiene valores por debajo del mínimo de negocio ({low})."
            )
        if high is not None and (series > high).any():
            raise DataValidationError(
                f"La columna '{col}' contiene valores por encima del máximo de negocio ({high})."
            )

    null_ratio = df[numeric_cols + categorical_cols].isna().mean()
    high_null_cols = null_ratio[null_ratio > 0.5]
    if not high_null_cols.empty:
        logger.warning(
            "Columnas con más de 50%% de nulos (se imputarán en el pipeline): %s",
            high_null_cols.to_dict(),
        )

    logger.info("Validación de esquema crudo superada correctamente.")


def validate_processed_schema(df: pd.DataFrame, target_col: str) -> None:
    """Valida un split ya procesado (train/validation/test) antes de usarlo para modelar."""
    logger.info("Validando esquema de datos procesados (%d filas, %d columnas)", *df.shape)

    if target_col not in df.columns:
        raise DataValidationError(f"El split procesado no contiene la columna objetivo '{target_col}'.")

    if df[target_col].isna().any():
        raise DataValidationError(
            f"La columna objetivo '{target_col}' contiene valores nulos tras el preprocesamiento."
        )

    unique_targets = set(df[target_col].unique())
    if not unique_targets.issubset({0, 1}):
        raise DataValidationError(
            f"La columna objetivo '{target_col}' debe ser binaria (0/1), valores encontrados: {unique_targets}"
        )

    logger.info("Validación de esquema procesado superada correctamente.")
