"""Carga y limpieza de datos crudos + transformador de preprocesamiento reutilizable.

Contiene:
- `load_raw_data`: lectura robusta del CSV crudo.
- `clean_target`: conversión de la variable objetivo (y de FLAG_UNICEF) de
  "vacío = no aplica" a binario 0/1, tal como se hizo en el notebook exploratorio.
- `OutlierCapper`: transformador sklearn-compatible que reemplaza el capping
  manual por percentil del notebook. Al ser un `TransformerMixin`, sus
  umbrales se *ajustan solo con el train* (evitando fuga de datos) y viajan
  dentro del mismo Pipeline que se serializa para servir el modelo.
- `build_column_transformer`: imputación (mediana/moda) + One-Hot Encoding,
  reemplazando el `pd.get_dummies` manual del notebook por un
  `ColumnTransformer` reproducible y apto para inferencia en producción.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.exceptions import DataValidationError

logger = logging.getLogger(__name__)


def load_raw_data(path: str, sep: str = ";", encoding: str = "utf-8") -> pd.DataFrame:
    """Lee el CSV crudo de seguros antirrobos.

    Lanza FileNotFoundError si la ruta no existe y DataValidationError si el
    archivo está vacío o no se puede parsear.
    """
    logger.info("Cargando dataset crudo desde %s", path)
    try:
        df = pd.read_csv(path, sep=sep, encoding=encoding)
    except FileNotFoundError:
        logger.error("No se encontró el archivo de datos crudos: %s", path)
        raise
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"No se pudo parsear el CSV crudo '{path}': {exc}") from exc

    if df.empty:
        raise DataValidationError(f"El archivo '{path}' se cargó pero no contiene filas.")

    logger.info("Dataset crudo cargado: %d filas, %d columnas", *df.shape)
    return df


def clean_target(df: pd.DataFrame, target_col: str, flag_unicef_col: str = "FLAG_UNICEF") -> pd.DataFrame:
    """Convierte FLAG_SS (y FLAG_UNICEF) de "1 o vacío" a binario 0/1.

    En el dataset crudo estas columnas solo registran el valor 1 cuando el
    evento ocurrió; la ausencia de compra/adhesión se representa como NaN.
    Esto es una decisión de negocio (no una imputación estadística), por lo
    que vive en un paso explícito y no dentro del ColumnTransformer.
    """
    df = df.copy()
    for col in (target_col, flag_unicef_col):
        if col not in df.columns:
            continue
        before_na = int(df[col].isna().sum())
        df[col] = df[col].fillna(0).astype(int)
        logger.info("Columna '%s': %d valores nulos re-etiquetados como 0.", col, before_na)
    return df


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Recorta (winsoriza) columnas numéricas al percentil `quantile` aprendido en fit.

    Reemplaza el capping manual del notebook (calculado sobre todo el dataset
    antes del split) por una versión que solo aprende del set de entrenamiento,
    evitando fuga de información hacia validación/test/producción.
    """

    def __init__(self, columns: list[str], quantile: float = 0.99):
        self.columns = columns
        self.quantile = quantile

    def fit(self, X: pd.DataFrame, y=None) -> "OutlierCapper":
        self.thresholds_: dict[str, float] = {
            col: float(X[col].dropna().quantile(self.quantile))
            for col in self.columns
            if col in X.columns
        }
        logger.info("OutlierCapper ajustado con umbrales p%.0f: %s", self.quantile * 100, self.thresholds_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, threshold in self.thresholds_.items():
            if col in X.columns:
                X[col] = np.where(X[col] > threshold, threshold, X[col])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def build_column_transformer(
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_impute_strategy: str = "median",
    categorical_impute_strategy: str = "most_frequent",
) -> ColumnTransformer:
    """Construye el ColumnTransformer de imputación + codificación.

    Numéricas: imputación (mediana por defecto).
    Categóricas: imputación (moda por defecto) + One-Hot Encoding, con
    `handle_unknown="ignore"` para que categorías nuevas en producción no
    rompan la inferencia.
    """
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy=numeric_impute_strategy))])

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=categorical_impute_strategy)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    logger.debug(
        "ColumnTransformer construido: %d numéricas, %d categóricas.",
        len(numeric_cols),
        len(categorical_cols),
    )
    return column_transformer
