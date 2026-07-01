"""Carga centralizada de configuración (params.yaml y configs/config.yaml).

Ningún módulo del proyecto debe leer un YAML directamente ni hardcodear rutas:
todos pasan por las funciones de este archivo para que exista un único punto
de verdad sobre dónde viven los parámetros reproducibles (params.yaml) y la
configuración de infraestructura (configs/config.yaml).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "params.yaml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    return content or {}


def load_params(path: str | Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Carga los hiperparámetros y parámetros reproducibles del pipeline (params.yaml)."""
    return _load_yaml(Path(path))


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Carga la configuración de rutas e infraestructura (configs/config.yaml)."""
    return _load_yaml(Path(path))


def resolve_path(relative_path: str | Path) -> Path:
    """Resuelve una ruta relativa del proyecto contra la raíz del repositorio."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
