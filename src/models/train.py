"""Entrenamiento, comparación y selección del mejor modelo de propensión.

Reproduce la comparación del notebook (Decision Tree, Random Forest, XGBoost)
pero de forma parametrizada (todo viene de params.yaml), registrando cada
candidato en MLflow y seleccionando automáticamente el mejor según la
métrica principal definida en `evaluation.primary_metric` (recall de la
clase positiva), usando `evaluation.secondary_metric` (ROC-AUC) como
desempate.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exceptions import ModelTrainingError
from src.features.data_validation import validate_processed_schema
from src.features.feature_engineering import build_full_pipeline
from src.models.evaluate import (
    evaluate_pipeline,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)
from src.models.mlflow_tracking import (
    configure_mlflow,
    log_figure_artifact,
    log_metrics,
    log_params,
    log_sklearn_model,
    mlflow_run,
)
from src.models.save_model import save_metadata, save_pipeline

logger = logging.getLogger(__name__)

_MODEL_FACTORIES = {
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}


def _build_classifier(model_name: str, model_params: dict[str, Any]) -> Any:
    if model_name not in _MODEL_FACTORIES:
        raise ModelTrainingError(f"Modelo '{model_name}' no reconocido. Opciones: {list(_MODEL_FACTORIES)}")
    return _MODEL_FACTORIES[model_name](**model_params)


def train_candidate(
    model_name: str,
    model_params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    preprocessing_params: dict[str, Any],
    feature_engineering_params: dict[str, Any],
    resampling_params: dict[str, Any],
    figures_dir: str,
    registry_model_name: str,
) -> dict[str, Any]:
    """Entrena un candidato, lo evalúa en validación y registra todo en MLflow."""
    classifier = _build_classifier(model_name, model_params)

    pipeline = build_full_pipeline(
        classifier=classifier,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        outlier_quantile=preprocessing_params["outlier_cap_quantile"],
        numeric_impute_strategy=preprocessing_params["numeric_impute_strategy"],
        categorical_impute_strategy=preprocessing_params["categorical_impute_strategy"],
        select_k_best=feature_engineering_params["select_k_best"],
        score_func=feature_engineering_params["score_func"],
        smote_random_state=resampling_params["random_state"],
    )

    logger.info("Entrenando candidato '%s'...", model_name)
    try:
        pipeline.fit(X_train, y_train)
    except Exception as exc:  # noqa: BLE001 - se relanza como error de dominio
        raise ModelTrainingError(f"Fallo al entrenar el modelo '{model_name}': {exc}") from exc

    result = evaluate_pipeline(pipeline, X_val, y_val)
    metrics = result["metrics"]
    logger.info("Métricas de validación para '%s': %s", model_name, metrics)

    with mlflow_run(run_name=model_name, tags={"stage": "model_selection"}):
        log_params({f"model__{k}": v for k, v in model_params.items()})
        log_params({"model_name": model_name})
        log_metrics(metrics)

        cm_path = f"{figures_dir}/confusion_matrix_{model_name}.png"
        roc_path = f"{figures_dir}/roc_curve_{model_name}.png"
        fi_path = f"{figures_dir}/feature_importance_{model_name}.png"

        plot_confusion_matrix(y_val.to_numpy(), result["y_pred"], cm_path, f"Matriz de Confusión - {model_name} (Validation)")
        plot_roc_curve(y_val.to_numpy(), result["y_proba"], roc_path, f"Curva ROC - {model_name} (Validation)")
        plot_feature_importance(pipeline, fi_path, f"Feature Importance - {model_name}")

        log_figure_artifact(cm_path)
        log_figure_artifact(roc_path)
        log_figure_artifact(fi_path)

        log_sklearn_model(pipeline, artifact_path="model")

    return {"model_name": model_name, "pipeline": pipeline, "metrics": metrics}


def select_best_candidate(
    candidates: list[dict[str, Any]], primary_metric: str, secondary_metric: str
) -> dict[str, Any]:
    """Selecciona el mejor candidato por métrica principal, con desempate por la secundaria."""
    best = max(
        candidates,
        key=lambda c: (c["metrics"][primary_metric], c["metrics"][secondary_metric]),
    )
    logger.info(
        "Modelo seleccionado: %s (%s=%.4f, %s=%.4f)",
        best["model_name"],
        primary_metric,
        best["metrics"][primary_metric],
        secondary_metric,
        best["metrics"][secondary_metric],
    )
    return best


def run_training_pipeline(
    train_path: str,
    validation_path: str,
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    preprocessing_params: dict[str, Any],
    feature_engineering_params: dict[str, Any],
    resampling_params: dict[str, Any],
    model_specs: dict[str, dict[str, Any]],
    evaluation_params: dict[str, Any],
    mlflow_params: dict[str, Any],
    model_output_path: str,
    model_metadata_path: str,
    train_metrics_output_path: str,
    figures_dir: str,
) -> dict[str, Any]:
    """Etapa `train` del pipeline DVC: entrena todos los candidatos y persiste el mejor."""
    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    validate_processed_schema(train_df, target_col=target_col)
    validate_processed_schema(validation_df, target_col=target_col)

    feature_cols = numeric_cols + categorical_cols
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = validation_df[feature_cols], validation_df[target_col]

    configure_mlflow(mlflow_params["tracking_uri"], mlflow_params["experiment_name"])

    candidates = []
    for model_name, model_params in model_specs.items():
        candidate = train_candidate(
            model_name=model_name,
            model_params=model_params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            preprocessing_params=preprocessing_params,
            feature_engineering_params=feature_engineering_params,
            resampling_params=resampling_params,
            figures_dir=figures_dir,
            registry_model_name=mlflow_params["registry_model_name"],
        )
        candidates.append(candidate)

    best = select_best_candidate(
        candidates,
        primary_metric=evaluation_params["primary_metric"],
        secondary_metric=evaluation_params["secondary_metric"],
    )

    save_pipeline(best["pipeline"], model_output_path)
    save_metadata(
        {
            "model_name": best["model_name"],
            "metrics_validation": best["metrics"],
            "all_candidates_metrics": {c["model_name"]: c["metrics"] for c in candidates},
            "feature_columns": feature_cols,
            "target_column": target_col,
            "primary_metric": evaluation_params["primary_metric"],
        },
        model_metadata_path,
    )

    import json
    from pathlib import Path

    Path(train_metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(train_metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"selected_model": best["model_name"], **{f"{c['model_name']}_{k}": v for c in candidates for k, v in c["metrics"].items()}},
            f,
            indent=2,
        )

    logger.info("Entrenamiento finalizado. Mejor modelo: %s", best["model_name"])
    return best
