"""Excepciones personalizadas del pipeline de Seguro Antirrobos.

Usar excepciones específicas (en vez de Exception genérica) permite que
main.py, la API y los tests distingan errores de negocio de errores de
programación, y devuelvan mensajes claros en cada capa.
"""


class SeguroAntirobosError(Exception):
    """Excepción base de todo el proyecto."""


class DataValidationError(SeguroAntirobosError):
    """Se lanza cuando el dataset crudo o procesado no cumple el esquema esperado."""


class DataSplitError(SeguroAntirobosError):
    """Se lanza cuando falla la división train/validation/test."""


class ModelTrainingError(SeguroAntirobosError):
    """Se lanza cuando el entrenamiento de un modelo falla."""


class ModelNotFoundError(SeguroAntirobosError):
    """Se lanza cuando no existe un modelo entrenado y persistido para cargar."""


class PredictionError(SeguroAntirobosError):
    """Se lanza cuando falla la generación de predicciones sobre datos nuevos."""
