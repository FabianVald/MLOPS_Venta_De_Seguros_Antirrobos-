"""Endpoints de la API de scoring de propensión al Seguro Antirrobos."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from src.exceptions import PredictionError
from src.models.predict import predict_batch, predict_single
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health(request: Request) -> HealthResponse:
    """Endpoint de salud para balanceadores/orquestadores (Docker/K8s)."""
    app_state = request.app.state
    model_loaded = getattr(app_state, "pipeline", None) is not None
    model_name = getattr(app_state, "model_name", None)
    return HealthResponse(status="ok", model_loaded=model_loaded, model_name=model_name)


@router.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Predice la probabilidad de que un cliente acepte el Seguro Antirrobos."""
    app_state = request.app.state
    if getattr(app_state, "pipeline", None) is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado todavía.")

    try:
        result = predict_single(
            pipeline=app_state.pipeline,
            record=payload.model_dump(),
            feature_columns=app_state.feature_columns,
            high_threshold=app_state.high_threshold,
            medium_threshold=app_state.medium_threshold,
        )
    except PredictionError as exc:
        logger.error("Error de predicción: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResponse(**result)


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["scoring"])
def predict_batch_endpoint(payload: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    """Predice la propensión de compra para un lote de clientes."""
    app_state = request.app.state
    if getattr(app_state, "pipeline", None) is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado todavía.")

    import pandas as pd

    try:
        df = pd.DataFrame([r.model_dump() for r in payload.records])
        result_df = predict_batch(
            pipeline=app_state.pipeline,
            df=df,
            feature_columns=app_state.feature_columns,
            high_threshold=app_state.high_threshold,
            medium_threshold=app_state.medium_threshold,
        )
    except PredictionError as exc:
        logger.error("Error de predicción batch: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    predictions = [
        PredictionResponse(
            prediction=int(row["prediction"]),
            probability=float(row["probability"]),
            propensity_band=str(row["propensity_band"]),
        )
        for _, row in result_df.iterrows()
    ]
    return BatchPredictionResponse(predictions=predictions)
