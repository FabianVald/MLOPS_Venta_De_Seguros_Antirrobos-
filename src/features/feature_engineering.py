"""Selección de variables y ensamblado del pipeline de features completo.

`build_feature_selector` encapsula el `SelectKBest` usado en el notebook.
`build_full_pipeline` ensambla, en un único `imblearn.pipeline.Pipeline`,
todos los pasos que deben viajar juntos desde datos crudos hasta la
predicción de un modelo:

    OutlierCapper -> ColumnTransformer (impute+encode) -> SelectKBest -> SMOTE -> Clasificador

Usar `imblearn.pipeline.Pipeline` (en vez de `sklearn.pipeline.Pipeline`) es
la pieza clave: SMOTE solo se aplica durante `.fit()` (para balancear el
entrenamiento); durante `.predict()`/`.predict_proba()` el paso de resampling
se omite automáticamente. Esto permite persistir un único artefacto con
`save_model.py` y cargarlo en la API sin tener que reimplementar el
balanceo en el servicio de inferencia.
"""

from __future__ import annotations

import logging

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import ClassifierMixin
from sklearn.feature_selection import SelectKBest, f_classif

from src.features.preprocessing import OutlierCapper, build_column_transformer

logger = logging.getLogger(__name__)

_SCORE_FUNCS = {"f_classif": f_classif}


def build_feature_selector(k: int, score_func: str = "f_classif") -> SelectKBest:
    """Construye el selector de variables SelectKBest usado tras la codificación."""
    if score_func not in _SCORE_FUNCS:
        raise ValueError(f"score_func '{score_func}' no soportado. Opciones: {list(_SCORE_FUNCS)}")
    logger.debug("SelectKBest construido con k=%d, score_func=%s", k, score_func)
    return SelectKBest(score_func=_SCORE_FUNCS[score_func], k=k)


def build_full_pipeline(
    classifier: ClassifierMixin,
    numeric_cols: list[str],
    categorical_cols: list[str],
    outlier_quantile: float,
    numeric_impute_strategy: str,
    categorical_impute_strategy: str,
    select_k_best: int,
    score_func: str,
    smote_random_state: int,
) -> ImbPipeline:
    """Ensambla el pipeline completo (preprocesamiento + selección + balanceo + modelo)."""
    outlier_capper = OutlierCapper(columns=numeric_cols, quantile=outlier_quantile)
    column_transformer = build_column_transformer(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_impute_strategy=numeric_impute_strategy,
        categorical_impute_strategy=categorical_impute_strategy,
    )
    feature_selector = build_feature_selector(k=select_k_best, score_func=score_func)
    smote = SMOTE(random_state=smote_random_state)

    pipeline = ImbPipeline(
        steps=[
            ("outlier_capper", outlier_capper),
            ("column_transformer", column_transformer),
            ("feature_selector", feature_selector),
            ("smote", smote),
            ("classifier", classifier),
        ]
    )
    logger.info("Pipeline completo ensamblado con clasificador %s", type(classifier).__name__)
    return pipeline
