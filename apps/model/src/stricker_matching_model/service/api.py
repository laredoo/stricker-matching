"""Minimal FastAPI surface for model inference."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from stricker_matching_model.core.artifacts import ArtifactStore
from stricker_matching_model.inference.predictor import Predictor

app = FastAPI(title="Stricker Model Service")


class PredictRequest(BaseModel):
    rows: List[List[float]]


class PredictResponse(BaseModel):
    labels: List[int]


@lru_cache
def _predictor() -> Predictor:
    artifact_path = Path("artifacts/model.joblib")
    return Predictor(artifacts=ArtifactStore(artifact_path))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "Model service is healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        labels = _predictor().predict(req.rows)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PredictResponse(labels=labels)
