import json
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, create_model

MODELS = Path(__file__).resolve().parent.parent / "models"

model = joblib.load(MODELS / "lgbm.pkl")
meta = json.loads((MODELS / "meta.json").read_text())
FEATURES = meta["features"]
THRESHOLD = meta["threshold"]

# Build the input schema from the saved feature list: every feature is a required float
Transaction = create_model("Transaction", **{name: (float, ...) for name in FEATURES})


class Prediction(BaseModel):
    fraud_score: float
    flagged: bool
    threshold: float
    latency_ms: float


app = FastAPI(title="Fraud Detection API")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(FEATURES), "threshold": THRESHOLD}


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction):
    start = time.perf_counter()
    row = pd.DataFrame([tx.model_dump()], columns=FEATURES)
    score = float(model.predict_proba(row)[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000
    return Prediction(
        fraud_score=score,
        flagged=score >= THRESHOLD,
        threshold=THRESHOLD,
        latency_ms=round(latency_ms, 2),
    )
