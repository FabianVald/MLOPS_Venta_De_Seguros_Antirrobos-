"""Cálculo de métricas de clasificación. Funciones puras, sin efectos secundarios.

Se centraliza aquí para que `train.py`, `evaluate.py` y los tests usen
exactamente la misma definición de cada métrica.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Devuelve accuracy, precision, recall, f1 y roc_auc para la clase positiva."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def get_classification_report_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(y_true, y_pred, zero_division=0)


def get_roc_curve_points(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return roc_curve(y_true, y_proba)


def get_precision_recall_curve_points(
    y_true: np.ndarray, y_proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return precision_recall_curve(y_true, y_proba)
