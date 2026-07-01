import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.exceptions import DataValidationError
from src.features.data_validation import validate_processed_schema, validate_raw_schema
from src.features.feature_engineering import build_feature_selector, build_full_pipeline

NUMERIC_COLS = ["Mto_TC", "SUELDO_ESTIMADO", "EDAD", "ANTIGUEDAD_MES"]
CATEGORICAL_COLS = ["MARCA", "Nombre_territorio", "FLAG_LIMA_PROVINCIA", "REGION", "SEXO", "SEGMENTO", "FLAG_UNICEF"]


def test_validate_raw_schema_raises_when_column_missing(synthetic_raw_df):
    df_missing = synthetic_raw_df.drop(columns=["EDAD"])
    with pytest.raises(DataValidationError):
        validate_raw_schema(df_missing, NUMERIC_COLS, CATEGORICAL_COLS, target_col="FLAG_SS")


def test_validate_raw_schema_raises_for_invalid_age_range(synthetic_raw_df):
    df_bad = synthetic_raw_df.copy()
    df_bad.loc[0, "EDAD"] = -5
    with pytest.raises(DataValidationError):
        validate_raw_schema(df_bad, NUMERIC_COLS, CATEGORICAL_COLS, target_col="FLAG_SS")


def test_validate_raw_schema_passes_for_clean_data(synthetic_raw_df):
    validate_raw_schema(synthetic_raw_df, NUMERIC_COLS, CATEGORICAL_COLS, target_col="FLAG_SS")


def test_validate_processed_schema_raises_for_non_binary_target(synthetic_processed_df):
    df_bad = synthetic_processed_df.copy()
    df_bad.loc[0, "FLAG_SS"] = 2
    with pytest.raises(DataValidationError):
        validate_processed_schema(df_bad, target_col="FLAG_SS")


def test_validate_processed_schema_raises_for_null_target(synthetic_processed_df):
    df_bad = synthetic_processed_df.copy()
    df_bad.loc[0, "FLAG_SS"] = None
    with pytest.raises(DataValidationError):
        validate_processed_schema(df_bad, target_col="FLAG_SS")


def test_build_feature_selector_invalid_score_func_raises():
    with pytest.raises(ValueError):
        build_feature_selector(k=5, score_func="no_existe")


def test_build_full_pipeline_has_expected_steps():
    pipeline = build_full_pipeline(
        classifier=DecisionTreeClassifier(max_depth=3, random_state=42),
        numeric_cols=NUMERIC_COLS,
        categorical_cols=CATEGORICAL_COLS,
        outlier_quantile=0.99,
        numeric_impute_strategy="median",
        categorical_impute_strategy="most_frequent",
        select_k_best=5,
        score_func="f_classif",
        smote_random_state=42,
    )
    assert list(pipeline.named_steps.keys()) == [
        "outlier_capper",
        "column_transformer",
        "feature_selector",
        "smote",
        "classifier",
    ]


def test_full_pipeline_fits_and_predicts_on_synthetic_data(synthetic_processed_df):
    pipeline = build_full_pipeline(
        classifier=DecisionTreeClassifier(max_depth=3, random_state=42),
        numeric_cols=NUMERIC_COLS,
        categorical_cols=CATEGORICAL_COLS,
        outlier_quantile=0.99,
        numeric_impute_strategy="median",
        categorical_impute_strategy="most_frequent",
        select_k_best=5,
        score_func="f_classif",
        smote_random_state=42,
    )
    X = synthetic_processed_df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = synthetic_processed_df["FLAG_SS"]

    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})
