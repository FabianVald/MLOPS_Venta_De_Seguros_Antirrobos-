"""Orquestador de línea de comandos del pipeline MLOps de Seguro Antirrobos.

Cada subcomando corresponde a una etapa de `dvc.yaml` y puede ejecutarse de
forma independiente para depuración:

    python main.py validate
    python main.py split
    python main.py train
    python main.py evaluate
    python main.py predict --input data/raw/DS_Seguro_Antirobos.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import load_config, load_params, resolve_path
from src.exceptions import SeguroAntirobosError
from src.features.data_validation import validate_raw_schema
from src.features.preprocessing import load_raw_data
from src.features.split_data import run_split_pipeline
from src.logging_setup import setup_logging
from src.models.evaluate import run_evaluation_on_test
from src.models.predict import load_model_for_inference, predict_batch
from src.models.train import run_training_pipeline

setup_logging()
logger = logging.getLogger(__name__)


def cmd_validate(_: argparse.Namespace) -> None:
    params = load_params()
    df = load_raw_data(
        str(resolve_path(params["data"]["raw_path"])),
        sep=params["data"]["sep"],
        encoding=params["data"]["encoding"],
    )
    validate_raw_schema(
        df,
        numeric_cols=params["columns"]["numeric"],
        categorical_cols=params["columns"]["categorical"],
        target_col=params["target"]["column"],
    )
    logger.info("Etapa 'validate' completada sin errores.")


def cmd_split(_: argparse.Namespace) -> None:
    params = load_params()
    run_split_pipeline(
        raw_path=str(resolve_path(params["data"]["raw_path"])),
        processed_dir=str(resolve_path(params["data"]["processed_dir"])),
        train_path=str(resolve_path(params["data"]["train_path"])),
        validation_path=str(resolve_path(params["data"]["validation_path"])),
        test_path=str(resolve_path(params["data"]["test_path"])),
        sep=params["data"]["sep"],
        encoding=params["data"]["encoding"],
        numeric_cols=params["columns"]["numeric"],
        categorical_cols=params["columns"]["categorical"],
        target_col=params["target"]["column"],
        test_size=params["data"]["test_size"],
        validation_size=params["data"]["validation_size"],
        random_state=params["data"]["random_state"],
    )
    logger.info("Etapa 'split' completada sin errores.")


def cmd_train(_: argparse.Namespace) -> None:
    params = load_params()
    config = load_config()

    run_training_pipeline(
        train_path=str(resolve_path(params["data"]["train_path"])),
        validation_path=str(resolve_path(params["data"]["validation_path"])),
        target_col=params["target"]["column"],
        numeric_cols=params["columns"]["numeric"],
        categorical_cols=params["columns"]["categorical"],
        preprocessing_params=params["preprocessing"],
        feature_engineering_params=params["feature_engineering"],
        resampling_params=params["resampling"],
        model_specs=params["models"],
        evaluation_params=params["evaluation"],
        mlflow_params=params["mlflow"],
        model_output_path=str(resolve_path(config["paths"]["model_file"])),
        model_metadata_path=str(resolve_path(config["paths"]["model_metadata_file"])),
        train_metrics_output_path=str(resolve_path(config["paths"]["reports_dir"])) + "/train_metrics.json",
        figures_dir=str(resolve_path(config["paths"]["figures_dir"])) + "/validation",
    )
    logger.info("Etapa 'train' completada sin errores.")


def cmd_evaluate(_: argparse.Namespace) -> None:
    params = load_params()
    config = load_config()

    run_evaluation_on_test(
        model_path=str(resolve_path(config["paths"]["model_file"])),
        test_path=str(resolve_path(params["data"]["test_path"])),
        target_col=params["target"]["column"],
        numeric_cols=params["columns"]["numeric"],
        categorical_cols=params["columns"]["categorical"],
        figures_dir=str(resolve_path(config["paths"]["figures_dir"])) + "/test",
        metrics_output_path=str(resolve_path(config["paths"]["reports_dir"])) + "/test_metrics.json",
    )
    logger.info("Etapa 'evaluate' completada sin errores.")


def cmd_predict(args: argparse.Namespace) -> None:
    import pandas as pd

    params = load_params()
    config = load_config()

    pipeline = load_model_for_inference(str(resolve_path(config["paths"]["model_file"])))
    df = pd.read_csv(args.input, sep=params["data"]["sep"] if args.input.endswith(".csv") and args.raw else ",")

    feature_cols = params["columns"]["numeric"] + params["columns"]["categorical"]
    result_df = predict_batch(
        pipeline=pipeline,
        df=df,
        feature_columns=feature_cols,
        high_threshold=params["evaluation"]["scoring_bands"]["high"],
        medium_threshold=params["evaluation"]["scoring_bands"]["medium"],
    )
    result_df.to_csv(args.output, index=False)
    logger.info("Predicciones guardadas en %s", args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline MLOps de Seguro Antirrobos")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="Valida el esquema del dataset crudo").set_defaults(func=cmd_validate)
    subparsers.add_parser("split", help="Divide el dataset en train/validation/test").set_defaults(func=cmd_split)
    subparsers.add_parser("train", help="Entrena, compara y persiste el mejor modelo").set_defaults(func=cmd_train)
    subparsers.add_parser("evaluate", help="Evalúa el modelo final sobre el test set").set_defaults(func=cmd_evaluate)

    predict_parser = subparsers.add_parser("predict", help="Genera predicciones batch sobre un CSV")
    predict_parser.add_argument("--input", required=True, help="Ruta al CSV de entrada")
    predict_parser.add_argument("--output", required=True, help="Ruta al CSV de salida con predicciones")
    predict_parser.add_argument("--raw", action="store_true", help="Indica que el CSV usa el separador ';' del dataset crudo")
    predict_parser.set_defaults(func=cmd_predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except SeguroAntirobosError as exc:
        logger.error("Error de dominio en la etapa '%s': %s", args.command, exc)
        sys.exit(1)
    except Exception:  # noqa: BLE001
        logger.exception("Error inesperado en la etapa '%s'", args.command)
        sys.exit(1)


if __name__ == "__main__":
    main()
