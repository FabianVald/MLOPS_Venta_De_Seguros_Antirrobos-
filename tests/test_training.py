import json

import pandas as pd
import pytest

from src.exceptions import ModelTrainingError
from src.models.train import _build_classifier, run_training_pipeline, select_best_candidate

NUMERIC_COLS = ["Mto_TC", "SUELDO_ESTIMADO", "EDAD", "ANTIGUEDAD_MES"]
CATEGORICAL_COLS = ["MARCA", "Nombre_territorio", "FLAG_LIMA_PROVINCIA", "REGION", "SEXO", "SEGMENTO", "FLAG_UNICEF"]
TARGET_COL = "FLAG_SS"

PREPROCESSING_PARAMS = {
    "outlier_cap_quantile": 0.99,
    "numeric_impute_strategy": "median",
    "categorical_impute_strategy": "most_frequent",
}
FEATURE_ENGINEERING_PARAMS = {"select_k_best": 5, "score_func": "f_classif"}
RESAMPLING_PARAMS = {"random_state": 42}
EVALUATION_PARAMS = {"primary_metric": "recall", "secondary_metric": "roc_auc"}

SMALL_MODEL_SPECS = {
    "decision_tree": {"max_depth": 3, "random_state": 42},
    "random_forest": {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1},
}


def test_build_classifier_raises_for_unknown_model():
    with pytest.raises(ModelTrainingError):
        _build_classifier("modelo_inexistente", {})


def test_select_best_candidate_prefers_higher_primary_metric():
    candidates = [
        {"model_name": "a", "metrics": {"recall": 0.5, "roc_auc": 0.9}},
        {"model_name": "b", "metrics": {"recall": 0.8, "roc_auc": 0.7}},
    ]
    best = select_best_candidate(candidates, primary_metric="recall", secondary_metric="roc_auc")
    assert best["model_name"] == "b"


def test_run_training_pipeline_end_to_end(tmp_path, synthetic_processed_df):
    train_df = synthetic_processed_df.iloc[:200]
    validation_df = synthetic_processed_df.iloc[200:]

    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)

    model_output_path = tmp_path / "models" / "model.pkl"
    model_metadata_path = tmp_path / "models" / "model_metadata.json"
    train_metrics_output_path = tmp_path / "reports" / "train_metrics.json"
    figures_dir = tmp_path / "figures"

    mlflow_params = {
        "tracking_uri": str(tmp_path / "mlruns"),
        "experiment_name": "test_experiment",
        "registry_model_name": "TestModel",
    }

    best = run_training_pipeline(
        train_path=str(train_path),
        validation_path=str(validation_path),
        target_col=TARGET_COL,
        numeric_cols=NUMERIC_COLS,
        categorical_cols=CATEGORICAL_COLS,
        preprocessing_params=PREPROCESSING_PARAMS,
        feature_engineering_params=FEATURE_ENGINEERING_PARAMS,
        resampling_params=RESAMPLING_PARAMS,
        model_specs=SMALL_MODEL_SPECS,
        evaluation_params=EVALUATION_PARAMS,
        mlflow_params=mlflow_params,
        model_output_path=str(model_output_path),
        model_metadata_path=str(model_metadata_path),
        train_metrics_output_path=str(train_metrics_output_path),
        figures_dir=str(figures_dir),
    )

    assert model_output_path.exists()
    assert model_metadata_path.exists()
    assert train_metrics_output_path.exists()
    assert best["model_name"] in SMALL_MODEL_SPECS

    metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_name"] == best["model_name"]
    assert "recall" in metadata["metrics_validation"]
