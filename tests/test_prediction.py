import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.exceptions import PredictionError
from src.features.feature_engineering import build_full_pipeline
from src.models.predict import assign_propensity_band, predict_batch, predict_single

NUMERIC_COLS = ["Mto_TC", "SUELDO_ESTIMADO", "EDAD", "ANTIGUEDAD_MES"]
CATEGORICAL_COLS = ["MARCA", "Nombre_territorio", "FLAG_LIMA_PROVINCIA", "REGION", "SEXO", "SEGMENTO", "FLAG_UNICEF"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


@pytest.fixture
def fitted_pipeline(synthetic_processed_df):
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
    X = synthetic_processed_df[FEATURE_COLS]
    y = synthetic_processed_df["FLAG_SS"]
    pipeline.fit(X, y)
    return pipeline


def test_assign_propensity_band_boundaries():
    assert assign_propensity_band(0.9, high_threshold=0.7, medium_threshold=0.5) == "alta"
    assert assign_propensity_band(0.6, high_threshold=0.7, medium_threshold=0.5) == "media"
    assert assign_propensity_band(0.2, high_threshold=0.7, medium_threshold=0.5) == "baja"


def test_predict_batch_returns_expected_columns(fitted_pipeline, synthetic_processed_df):
    df = synthetic_processed_df[FEATURE_COLS].head(10)
    result = predict_batch(fitted_pipeline, df, FEATURE_COLS, high_threshold=0.7, medium_threshold=0.5)

    assert {"prediction", "probability", "propensity_band"}.issubset(result.columns)
    assert len(result) == 10
    assert set(result["propensity_band"]).issubset({"alta", "media", "baja"})


def test_predict_batch_raises_when_columns_missing(fitted_pipeline, synthetic_processed_df):
    df = synthetic_processed_df[FEATURE_COLS].drop(columns=["EDAD"]).head(5)
    with pytest.raises(PredictionError):
        predict_batch(fitted_pipeline, df, FEATURE_COLS, high_threshold=0.7, medium_threshold=0.5)


def test_predict_single_returns_expected_keys(fitted_pipeline, synthetic_processed_df):
    record = synthetic_processed_df[FEATURE_COLS].iloc[0].to_dict()
    result = predict_single(fitted_pipeline, record, FEATURE_COLS, high_threshold=0.7, medium_threshold=0.5)

    assert set(result.keys()) == {"prediction", "probability", "propensity_band"}
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0


def test_health_endpoint_without_model(monkeypatch, tmp_path):
    """La API debe arrancar y responder /health incluso sin un modelo entrenado."""
    from fastapi.testclient import TestClient

    from src.config import PROJECT_ROOT

    monkeypatch.chdir(PROJECT_ROOT)
    from src.api.app import app

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
