"""Punto de entrada de la API FastAPI de scoring del Seguro Antirrobos.

Ejecutar con: uvicorn src.api.app:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import router
from src.config import load_config, load_params, resolve_path
from src.exceptions import ModelNotFoundError
from src.logging_setup import setup_logging
from src.models.predict import load_model_for_inference
from src.models.save_model import load_metadata

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo entrenado una sola vez, al iniciar el servicio."""
    config = load_config()
    params = load_params()

    model_path = resolve_path(config["api"]["model_path"])
    metadata_path = resolve_path(config["api"]["model_metadata_path"])

    app.state.feature_columns = params["columns"]["numeric"] + params["columns"]["categorical"]
    app.state.high_threshold = params["evaluation"]["scoring_bands"]["high"]
    app.state.medium_threshold = params["evaluation"]["scoring_bands"]["medium"]

    try:
        app.state.pipeline = load_model_for_inference(str(model_path))
        metadata = load_metadata(str(metadata_path))
        app.state.model_name = metadata.get("model_name")
        logger.info("Modelo '%s' cargado correctamente desde %s", app.state.model_name, model_path)
    except ModelNotFoundError as exc:
        logger.warning("La API arranca sin modelo cargado: %s", exc)
        app.state.pipeline = None
        app.state.model_name = None

    yield

    logger.info("Apagando la API de scoring.")


config = load_config()
app = FastAPI(
    title=config["api"]["title"],
    version=config["api"]["version"],
    lifespan=lifespan,
)
app.include_router(router)
