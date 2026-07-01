"""Evaluación de un pipeline entrenado: métricas + gráficas (matriz de confusión,
curva ROC, curva Precision-Recall, feature importance).

Se usa en dos momentos distintos:
1. Dentro de `train.py`, para comparar candidatos sobre el set de validación.
2. Como etapa independiente de DVC (`python main.py evaluate`), para medir
   el modelo ya seleccionado sobre el test set que nunca se tocó durante
   entrenamiento/selección de hiperparámetros.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # backend sin display, requerido para correr en CI/servidores
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.models.metrics import (
    compute_classification_metrics,
    get_confusion_matrix,
    get_precision_recall_curve_points,
    get_roc_curve_points,
)
from src.models.save_model import load_pipeline

logger = logging.getLogger(__name__)


def get_selected_feature_names(pipeline: Any) -> list[str]:
    """Recupera los nombres de las features que sobrevivieron a SelectKBest."""
    column_transformer = pipeline.named_steps["column_transformer"]
    feature_selector = pipeline.named_steps["feature_selector"]

    all_feature_names = column_transformer.get_feature_names_out()
    support_mask = feature_selector.get_support()
    return list(np.asarray(all_feature_names)[support_mask])


def evaluate_pipeline(pipeline: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Calcula predicciones y métricas de un pipeline ya entrenado sobre (X, y)."""
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    metrics = compute_classification_metrics(y.to_numpy(), y_pred, y_proba)
    return {"y_pred": y_pred, "y_proba": y_proba, "metrics": metrics}


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str, title: str) -> None:
    cm = get_confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No compra", "Compra"], yticklabels=["No compra", "Compra"])
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info("Matriz de confusión guardada en %s", output_path)


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, output_path: str, title: str) -> None:
    fpr, tpr, _ = get_roc_curve_points(y_true, y_proba)
    auc = compute_classification_metrics(y_true, (y_proba >= 0.5).astype(int), y_proba)["roc_auc"]
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info("Curva ROC guardada en %s", output_path)


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, output_path: str, title: str) -> None:
    precision, recall, _ = get_precision_recall_curve_points(y_true, y_proba)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info("Curva Precision-Recall guardada en %s", output_path)


def plot_feature_importance(pipeline: Any, output_path: str, title: str, top_n: int = 15) -> None:
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        logger.warning("El clasificador %s no expone feature_importances_; se omite el gráfico.", type(classifier).__name__)
        return

    feature_names = get_selected_feature_names(pipeline)
    importances = classifier.feature_importances_
    df_importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_importance, x="importance", y="feature")
    plt.title(title)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info("Feature importance guardada en %s", output_path)


def run_evaluation_on_test(
    model_path: str,
    test_path: str,
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    figures_dir: str,
    metrics_output_path: str,
) -> dict[str, Any]:
    """Etapa `evaluate` del pipeline DVC: mide el modelo final contra el test set."""
    pipeline = load_pipeline(model_path)
    test_df = pd.read_csv(test_path)

    feature_cols = numeric_cols + categorical_cols
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    result = evaluate_pipeline(pipeline, X_test, y_test)
    logger.info("Métricas en test: %s", result["metrics"])

    plot_confusion_matrix(
        y_test.to_numpy(), result["y_pred"], f"{figures_dir}/confusion_matrix_test.png", "Matriz de Confusión (Test)"
    )
    plot_roc_curve(y_test.to_numpy(), result["y_proba"], f"{figures_dir}/roc_curve_test.png", "Curva ROC (Test)")
    plot_precision_recall_curve(
        y_test.to_numpy(), result["y_proba"], f"{figures_dir}/precision_recall_test.png", "Curva Precision-Recall (Test)"
    )
    plot_feature_importance(pipeline, f"{figures_dir}/feature_importance_test.png", "Feature Importance (Test)")

    Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)
    logger.info("Métricas de test guardadas en %s", metrics_output_path)

    return result["metrics"]
