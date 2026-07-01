"""Configuración centralizada de logging para todo el proyecto.

Ningún módulo debe usar print(): siempre `logging.getLogger(__name__)` tras
llamar a `setup_logging()` una vez al inicio de cada punto de entrada
(main.py, src/api/app.py, tests).
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT

DEFAULT_LOGGING_CONFIG_PATH = PROJECT_ROOT / "configs" / "logging.yaml"
_CONFIGURED = False


def setup_logging(config_path: str | Path = DEFAULT_LOGGING_CONFIG_PATH) -> None:
    """Aplica la configuración de logging.yaml. Es idempotente: solo se aplica una vez."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    config_path = Path(config_path)
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        logging.basicConfig(level=logging.INFO)
        logging.warning(
            "No se encontró %s, usando logging.basicConfig por defecto.", config_path
        )
        _CONFIGURED = True
        return

    with open(config_path, "r", encoding="utf-8") as f:
        logging_config = yaml.safe_load(f)

    logging.config.dictConfig(logging_config)
    _CONFIGURED = True
