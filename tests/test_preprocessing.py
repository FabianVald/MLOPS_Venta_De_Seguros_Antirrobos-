import numpy as np
import pandas as pd
import pytest

from src.exceptions import DataValidationError
from src.features.preprocessing import (
    OutlierCapper,
    build_column_transformer,
    clean_target,
    load_raw_data,
)


def test_load_raw_data_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        load_raw_data("data/raw/no_existe.csv", sep=";")


def test_load_raw_data_raises_for_empty_file(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("Mto_TC;MARCA\n")
    with pytest.raises(DataValidationError):
        load_raw_data(str(empty_file), sep=";")


def test_clean_target_fills_nan_with_zero(synthetic_raw_df):
    cleaned = clean_target(synthetic_raw_df, target_col="FLAG_SS", flag_unicef_col="FLAG_UNICEF")
    assert cleaned["FLAG_SS"].isna().sum() == 0
    assert set(cleaned["FLAG_SS"].unique()).issubset({0, 1})
    assert cleaned["FLAG_UNICEF"].isna().sum() == 0


def test_outlier_capper_caps_values_above_learned_threshold():
    train = pd.DataFrame({"Mto_TC": [100, 200, 300, 400, 100000]})
    capper = OutlierCapper(columns=["Mto_TC"], quantile=0.99)
    capper.fit(train)

    new_data = pd.DataFrame({"Mto_TC": [50, 999999]})
    capped = capper.transform(new_data)

    assert capped["Mto_TC"].iloc[0] == 50  # valor bajo, no se toca
    assert capped["Mto_TC"].iloc[1] == pytest.approx(capper.thresholds_["Mto_TC"])


def test_outlier_capper_preserves_nan():
    train = pd.DataFrame({"EDAD": [20, 30, 40, 50, 60]})
    capper = OutlierCapper(columns=["EDAD"], quantile=0.99).fit(train)

    new_data = pd.DataFrame({"EDAD": [25, np.nan]})
    capped = capper.transform(new_data)

    assert pd.isna(capped["EDAD"].iloc[1])


def test_build_column_transformer_output_shape(synthetic_processed_df):
    numeric_cols = ["Mto_TC", "SUELDO_ESTIMADO", "EDAD", "ANTIGUEDAD_MES"]
    categorical_cols = ["MARCA", "Nombre_territorio", "FLAG_LIMA_PROVINCIA", "REGION", "SEXO", "SEGMENTO", "FLAG_UNICEF"]

    transformer = build_column_transformer(numeric_cols, categorical_cols)
    transformed = transformer.fit_transform(synthetic_processed_df[numeric_cols + categorical_cols])

    assert transformed.shape[0] == len(synthetic_processed_df)
    # no debe quedar ningún NaN tras imputar
    assert not np.isnan(transformed.toarray() if hasattr(transformed, "toarray") else transformed).any()
