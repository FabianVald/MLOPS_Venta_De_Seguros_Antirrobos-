"""Fixtures compartidas: generan un dataset sintético con la misma forma
(columnas y tipos) que DS_Seguro_Antirobos.csv, pero pequeño y rápido de
procesar, para no depender del CSV real de 28k filas en los tests unitarios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

NUMERIC_COLS = ["Mto_TC", "SUELDO_ESTIMADO", "EDAD", "ANTIGUEDAD_MES"]
CATEGORICAL_COLS = ["MARCA", "Nombre_territorio", "FLAG_LIMA_PROVINCIA", "REGION", "SEXO", "SEGMENTO", "FLAG_UNICEF"]
TARGET_COL = "FLAG_SS"


@pytest.fixture
def synthetic_raw_df() -> pd.DataFrame:
    """Simula el dataset crudo: target y FLAG_UNICEF con NaN representando "no aplica"."""
    rng = np.random.default_rng(42)
    n_rows = 300

    df = pd.DataFrame(
        {
            "Mto_TC": rng.integers(500, 20000, n_rows).astype(float),
            "MARCA": rng.choice(["Visa", "MasterCard"], n_rows, p=[0.95, 0.05]),
            "Nombre_territorio": rng.choice(["T.CENTRO", "T.NORTE", "T.SUR"], n_rows),
            "FLAG_LIMA_PROVINCIA": rng.choice([0.0, 1.0, np.nan], n_rows, p=[0.4, 0.55, 0.05]),
            "REGION": rng.choice(["CENTRO", "NORTE", "SUR", np.nan], n_rows, p=[0.3, 0.3, 0.35, 0.05]),
            "SUELDO_ESTIMADO": rng.uniform(900, 8000, n_rows),
            "EDAD": rng.integers(18, 70, n_rows).astype(float),
            "SEXO": rng.choice(["M", "F", np.nan], n_rows, p=[0.48, 0.48, 0.04]),
            "ANTIGUEDAD_MES": rng.integers(1, 200, n_rows).astype(float),
            "SEGMENTO": rng.choice(["CLASICO", "BAJO VALOR", "VIP", np.nan], n_rows, p=[0.6, 0.25, 0.1, 0.05]),
            "FLAG_SS": rng.choice([1.0, np.nan], n_rows, p=[0.2, 0.8]),
            "FLAG_UNICEF": rng.choice([1.0, np.nan], n_rows, p=[0.05, 0.95]),
        }
    )
    return df


@pytest.fixture
def synthetic_processed_df(synthetic_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Versión ya "limpia" del target, lista para dividir/entrenar."""
    df = synthetic_raw_df.copy()
    df[TARGET_COL] = df[TARGET_COL].fillna(0).astype(int)
    df["FLAG_UNICEF"] = df["FLAG_UNICEF"].fillna(0).astype(int)
    return df
