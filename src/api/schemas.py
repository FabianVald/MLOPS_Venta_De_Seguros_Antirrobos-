"""Contratos de entrada/salida de la API, validados con Pydantic.

Los campos opcionales reflejan columnas que en el dataset histórico tienen
nulos y que el pipeline sabe imputar (ver `src/features/preprocessing.py`);
no es necesario que el cliente de la API los conozca todos para pedir un score.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    Mto_TC: float = Field(..., ge=0, description="Monto de oferta de línea de tarjeta de crédito")
    MARCA: str = Field(..., description="Tipo de operador de la TC (Visa, MasterCard)")
    Nombre_territorio: str = Field(..., description="Territorio comercial del cliente")
    FLAG_LIMA_PROVINCIA: Optional[float] = Field(None, description="1 = Lima, 0 = Provincia")
    REGION: Optional[str] = Field(None, description="Región geográfica de residencia")
    SUELDO_ESTIMADO: Optional[float] = Field(None, ge=0, description="Sueldo estimado del cliente")
    EDAD: Optional[float] = Field(None, ge=18, le=120, description="Edad del cliente")
    SEXO: Optional[str] = Field(None, description="Sexo del cliente (M/F)")
    ANTIGUEDAD_MES: Optional[float] = Field(None, ge=0, description="Antigüedad del cliente en meses")
    SEGMENTO: Optional[str] = Field(None, description="Segmento comercial del cliente")
    FLAG_UNICEF: Optional[float] = Field(0, description="1 = cliente adherido a donación UNICEF")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Mto_TC": 5000,
                "MARCA": "Visa",
                "Nombre_territorio": "T.CENTRO",
                "FLAG_LIMA_PROVINCIA": 0,
                "REGION": "CENTRO",
                "SUELDO_ESTIMADO": 2500.0,
                "EDAD": 35,
                "SEXO": "M",
                "ANTIGUEDAD_MES": 48,
                "SEGMENTO": "CLASICO",
                "FLAG_UNICEF": 0,
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    records: list[PredictionRequest]


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = no compra, 1 = alta probabilidad de compra")
    probability: float = Field(..., description="Probabilidad estimada de compra del seguro")
    propensity_band: str = Field(..., description="Banda de propensión: alta, media o baja")


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
