"""
FastAPI service for real-time transaction anomaly/fraud scoring.

Endpoints:
  GET  /health         - liveness check
  POST /predict         - score a single transaction
  GET  /monitor/summary - recent prediction stats (for the monitoring script)

Run: uvicorn app.main:app --reload --port 8000
"""
import csv
import json
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = os.environ.get("MODEL_DIR", "C:\\Users\\Admin\\Downloads\\fraud-detection-project\\models")
LOG_PATH = os.environ.get("PREDICTION_LOG", "C:\\Users\\Admin\\Downloads\\fraud-detection-project\\data\\prediction_log.csv")

FEATURES = ["amount", "hour_of_day", "txn_count_last_hour",
            "distance_from_home_km", "is_new_merchant"]

app = FastAPI(title="Fraud/Anomaly Detection API", version="1.0.0")

model = joblib.load(f"{MODEL_DIR}/isolation_forest.joblib")
scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
with open(f"{MODEL_DIR}/baseline_stats.json") as f:
    BASELINE = json.load(f)


class Transaction(BaseModel):
    txn_id: str = Field(..., examples=["TXN123456"])
    amount: float = Field(..., ge=0)
    hour_of_day: float = Field(..., ge=0, le=23)
    txn_count_last_hour: int = Field(..., ge=0)
    distance_from_home_km: float = Field(..., ge=0)
    is_new_merchant: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    txn_id: str
    is_anomaly: bool
    anomaly_score: float
    threshold_p95: float
    latency_ms: float


def _ensure_log_header():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "txn_id", "anomaly_score", "is_anomaly"])


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: Transaction):
    start = time.perf_counter()
    try:
        x = np.array([[getattr(txn, f) for f in FEATURES]])
        x_scaled = scaler.transform(x)
        score = float(-model.score_samples(x_scaled)[0])  # higher = more anomalous
        is_anomaly = bool(score >= BASELINE["p95_score"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scoring failed: {e}")

    latency_ms = (time.perf_counter() - start) * 1000

    _ensure_log_header()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(timezone.utc).isoformat(), txn.txn_id, score, is_anomaly])

    return PredictionResponse(
        txn_id=txn.txn_id,
        is_anomaly=is_anomaly,
        anomaly_score=round(score, 4),
        threshold_p95=round(BASELINE["p95_score"], 4),
        latency_ms=round(latency_ms, 2),
    )


@app.get("/monitor/summary")
def monitor_summary():
    """Lightweight peek at recent predictions — what monitor.py polls."""
    if not os.path.exists(LOG_PATH):
        return {"count": 0}
    import pandas as pd
    df = pd.read_csv(LOG_PATH)
    if df.empty:
        return {"count": 0}
    return {
        "count": len(df),
        "recent_anomaly_rate": float(df["is_anomaly"].tail(200).mean()),
        "baseline_mean_score": BASELINE["mean_score"],
        "recent_mean_score": float(df["anomaly_score"].tail(200).mean()),
    }
