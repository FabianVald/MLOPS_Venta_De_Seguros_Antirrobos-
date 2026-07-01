"""División de datos crudos en train / validation / test, persistidos a CSV.

El notebook original solo hacía un split 70/30 (train/test). Para poder
seleccionar hiperparámetros y el mejor modelo sin tocar el test set (que se
reserva exclusivamente para la evaluación final, ver `src/models/evaluate.py`),
este módulo agrega un tercer split de validación, tal como exige la
estructura de `data/processed/` del proyecto.

La columna objetivo se limpia (NaN -> 0) *antes* de dividir: es una regla de
negocio fija, no una estadística aprendida, por lo que no hay fuga de datos
al aplicarla sobre el dataset completo.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exceptions import DataSplitError
from src.features.data_validation import validate_processed_schema, validate_raw_schema
from src.features.preprocessing import clean_target, load_raw_data

logger = logging.getLogger(__name__)


def split_train_validation_test(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide `df` en train/validation/test estratificados por `target_col`.

    `test_size` y `validation_size` son proporciones sobre el dataset completo.
    """
    if test_size + validation_size >= 1.0:
        raise DataSplitError(
            f"test_size ({test_size}) + validation_size ({validation_size}) debe ser < 1.0"
        )

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )

    # validation_size está expresado sobre el dataset completo; se reescala
    # sobre lo que queda tras separar el test set.
    remaining_fraction = 1.0 - test_size
    validation_fraction_of_remaining = validation_size / remaining_fraction

    train_df, validation_df = train_test_split(
        train_val_df,
        test_size=validation_fraction_of_remaining,
        random_state=random_state,
        stratify=train_val_df[target_col],
    )

    logger.info(
        "Split generado -> train: %d filas, validation: %d filas, test: %d filas",
        len(train_df),
        len(validation_df),
        len(test_df),
    )
    return train_df, validation_df, test_df


def run_split_pipeline(
    raw_path: str,
    processed_dir: str,
    train_path: str,
    validation_path: str,
    test_path: str,
    sep: str,
    encoding: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> None:
    """Pipeline completo: carga cruda -> valida -> limpia target -> divide -> persiste."""
    df = load_raw_data(raw_path, sep=sep, encoding=encoding)
    validate_raw_schema(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols, target_col=target_col)

    df = clean_target(df, target_col=target_col)

    train_df, validation_df, test_df = split_train_validation_test(
        df,
        target_col=target_col,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )

    for split_df in (train_df, validation_df, test_df):
        validate_processed_schema(split_df, target_col=target_col)

    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(
        "Splits persistidos en %s, %s, %s", train_path, validation_path, test_path
    )
