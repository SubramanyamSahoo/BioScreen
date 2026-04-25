"""
FastAPI server for production screening.

Run:
    uvicorn bioscreen.api.server:app --host 0.0.0.0 --port 8080
"""

import os
import time
import uuid
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

logger = logging.getLogger("bioscreen.api")

_screener = None
security = HTTPBearer(auto_error=False)


class ScreeningRequest(BaseModel):
    sequences: List[str] = Field(..., min_length=1, max_length=100)
    request_id: Optional[str] = None

    @validator("sequences", each_item=True)
    def validate_sequence(cls, v):
        if len(v) < 10:
            raise ValueError("Sequence must be at least 10 amino acids")
        if len(v) > 1500:
            raise ValueError("Sequence must be at most 1500 amino acids")
        return v.upper()


class ScreeningResponse(BaseModel):
    request_id: str
    results: List[dict]
    total_sequences: int
    processing_time_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_memory_gb: Optional[float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _screener
    import torch
    model_path = os.environ.get("BIOSCREEN_MODEL_PATH", "bioscreen-v1")
    from bioscreen import BioScreener
    _screener = BioScreener.from_pretrained(model_path)
    logger.info(f"Model loaded from {model_path}")
    yield
    del _screener
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="BioScreen API",
    description="Function-aware DNA synthesis screening against AI-designed biological threats",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    expected_key = os.environ.get("BIOSCREEN_API_KEY")
    if expected_key and (not credentials or credentials.credentials != expected_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials


@app.post("/v1/screen", response_model=ScreeningResponse)
async def screen_sequences(request: ScreeningRequest, _: str = Depends(get_api_key)):
    request_id = request.request_id or str(uuid.uuid4())
    t_start = time.perf_counter()
    try:
        results = _screener.screen(request.sequences, request_id=request_id)
        processing_time = (time.perf_counter() - t_start) * 1000
        return ScreeningResponse(
            request_id=request_id,
            results=[r.to_dict() for r in results],
            total_sequences=len(request.sequences),
            processing_time_ms=round(processing_time, 2),
            model_version="1.0.0",
        )
    except Exception as e:
        logger.exception("Screening error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/health", response_model=HealthResponse)
async def health_check():
    import torch
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return HealthResponse(
        status="healthy" if _screener is not None else "loading",
        model_loaded=_screener is not None,
        gpu_available=torch.cuda.is_available(),
        gpu_memory_gb=gpu_mem,
    )


@app.get("/v1/model/info")
async def model_info():
    return {
        "model": "bioscreen-v1",
        "version": "1.0.0",
        "architecture": "ESM-2 3B + contrastive functional embedding",
        "tasks": [
            "binary_threat_detection",
            "mechanism_of_harm_classification",
            "continuous_risk_scoring",
        ],
        "mechanisms": [
            "enzymatic_disruption", "hemolysis", "host_adhesion",
            "immune_evasion", "membrane_disruption", "neurotoxicity",
            "viral_entry", "benign",
        ],
        "num_threat_mechanisms": 7,
        "training_sequences": 4981,
        "max_sequence_length": 1500,
        "throughput_seq_per_sec": 44.4,
        "auroc": 0.998,
        "detection_at_40pct_identity": 0.993,
        "mechanism_accuracy": 0.908,
        "citation": "AIxBio Hackathon 2026, Track 1: DNA Screening & Synthesis Controls",
    }
