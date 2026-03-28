"""CLI and container entrypoint for the model service."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import uvicorn

from stricker_matching_model.core.artifacts import ArtifactStore
from stricker_matching_model.core.facade import ModelFacade
from stricker_matching_model.core.strategies import KMeansStrategy
from stricker_matching_model.etl.statsbomb import StatsBombETL
from stricker_matching_model.features.builder import FeatureBuilder
from stricker_matching_model.inference.predictor import Predictor
from stricker_matching_model.logging import configure_logging, get_logger
from stricker_matching_model.pipeline.builder import PipelineBuilder
from stricker_matching_model.train.trainer import StatsBombTrainer

logger = get_logger(__name__)


def _build_facade(
    artifact_path: Path,
    demo_data: bool,
    strategy_name: str,
    n_clusters: int,
) -> ModelFacade:
    etl = StatsBombETL(data_path=None, demo=demo_data)
    features = FeatureBuilder()
    pipeline_builder = PipelineBuilder()
    if strategy_name == "kmeans":
        strategy = KMeansStrategy(n_clusters=n_clusters)
    else:
        raise ValueError(f"Unsupported strategy: {strategy_name}")
    artifacts = ArtifactStore(artifact_path)
    trainer = StatsBombTrainer(
        etl=etl,
        features=features,
        pipeline_builder=pipeline_builder,
        strategy=strategy,
        artifacts=artifacts,
    )
    predictor = Predictor(artifacts=artifacts)
    return ModelFacade(trainer=trainer, predictor=predictor)


def _cmd_etl(args: argparse.Namespace) -> None:
    etl = StatsBombETL(
        data_path=Path(args.data_path),
        output_path=Path(args.output_path) if args.output_path else None,
        output_format=args.format,
        flip_left_to_right=args.flip_left_to_right,
        pitch_length=args.pitch_length,
        pitch_width=args.pitch_width,
        target_length=args.target_length,
        target_width=args.target_width,
        overwrite_files=args.overwrite_files,
    )
    files = etl.extract()
    normalized_stream = etl.transform(files)
    match_written = etl.load(normalized_stream)
    logger.info("Wrote %s normalized match event files", len(match_written))


def _cmd_train(args: argparse.Namespace) -> None:
    facade = _build_facade(
        Path(args.artifact_path),
        demo_data=args.demo,
        strategy_name=args.strategy,
        n_clusters=args.n_clusters,
    )
    facade.train()


def _cmd_predict(args: argparse.Namespace) -> None:
    facade = _build_facade(
        Path(args.artifact_path),
        demo_data=False,
        strategy_name="kmeans",
        n_clusters=3,
    )
    payload = json.loads(Path(args.input_json).read_text())
    rows = payload["rows"]
    labels = facade.predict(rows)
    output: dict[str, Any] = {"labels": labels}
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(output))
    else:
        print(json.dumps(output))


def _cmd_features(args: argparse.Namespace) -> None:
    builder = FeatureBuilder()
    data_path = Path(args.data_path)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else data_path / "processed" / "features" / "player_features.json"
    )
    result = builder.build(data_path, output_path)
    logger.info("Wrote feature file to %s", result)


def _cmd_server(args: argparse.Namespace) -> None:
    uvicorn.run(
        "stricker_matching_model.service.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stricker model CLI")
    parser.add_argument(
        "--log-level",
        default=os.getenv("STRICKER_LOG_LEVEL", "INFO"),
        help="Set logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    etl = sub.add_parser("etl", help="Normalize StatsBomb event files")
    etl.add_argument("--data-path", required=True)
    etl.add_argument("--output-path")
    etl.add_argument(
        "--format", choices=["json"], default="json"
    )  # include parquet in the future
    etl.add_argument("--flip-left-to-right", action="store_true")
    etl.add_argument("--pitch-length", type=float, default=120.0)
    etl.add_argument("--pitch-width", type=float, default=80.0)
    etl.add_argument("--target-length", type=float, default=105.0)
    etl.add_argument("--target-width", type=float, default=68.0)
    etl.add_argument("--overwrite-files", action="store_true")
    etl.set_defaults(func=_cmd_etl)

    train = sub.add_parser("train", help="Train and persist the model artifact")
    train.add_argument("--artifact-path", default="artifacts/model.joblib")
    train.add_argument("--demo", action="store_true", help="Use demo data")
    train.add_argument("--strategy", default="kmeans", choices=["kmeans"])
    train.add_argument("--n-clusters", type=int, default=3)
    train.set_defaults(func=_cmd_train)

    predict = sub.add_parser("predict", help="Predict using a saved artifact")
    predict.add_argument("--artifact-path", default="artifacts/model.joblib")
    predict.add_argument("--input-json", required=True)
    predict.add_argument("--output-json")
    predict.set_defaults(func=_cmd_predict)

    features = sub.add_parser("features", help="Build player feature dataset")
    features.add_argument("--data-path", required=True)
    features.add_argument("--output-path")
    features.set_defaults(func=_cmd_features)

    server = sub.add_parser("server", help="Start the model HTTP service")
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8001)
    server.add_argument("--reload", action="store_true")
    server.set_defaults(func=_cmd_server)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    args.func(args)


if __name__ == "__main__":
    main()
