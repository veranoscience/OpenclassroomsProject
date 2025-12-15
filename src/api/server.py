from __future__ import annotations

import os
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from huggingface_hub import hf_hub_download
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .schemas import PredictRequest, EmployeeInput  # pydantic v2

# -----------------------
# Config & chargement ML
# -----------------------
THRESHOLD = 0.33

LOCAL_MODEL = Path("models/model.joblib")
HF_REPO_ID = "veranoscience/attrition-model"
HF_FILENAME = "model.joblib"

def load_pipeline():
    """Charge le pipeline sklearn (préproc + RF) depuis /models sinon depuis le Hub."""
    if LOCAL_MODEL.exists():
        return joblib.load(LOCAL_MODEL)
    downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    return joblib.load(downloaded)

pipe = load_pipeline()

# -----------------------
# DB (optionnelle)
# -----------------------
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
db_engine = create_engine(DATABASE_URL, future=True) if DATABASE_URL else None

def log_prediction(source: str, inputs: dict | list, proba: float | None, pred: int | None,
                   threshold: float, model_version: str = "rf_reg@hub",
                   status: str = "ok", error_message: str | None = None) -> None:
    """Écrit une ligne dans ml.predictions_log si db_engine est dispo. Nève pas d'exception."""
    if not db_engine:
        # Pas de DB → on sort silencieusement
        return
    try:
        payload = json.dumps(inputs, ensure_ascii=False)
        with db_engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO ml.predictions_log
                      (request_id, source, input_payload, proba, pred, threshold, model_version, status, error_message)
                    VALUES
                      (:rid, :src, CAST(:payload AS JSONB), :proba, :pred, :thr, :version, :status, :err)
                """),
                {
                    "rid": str(uuid.uuid4()),
                    "src": source,
                    "payload": payload,
                    "proba": proba,
                    "pred": pred,
                    "thr": threshold,
                    "version": model_version,
                    "status": status,
                    "err": error_message
                }
            )
    except Exception as e:
        # On évite de casser l'API pour un souci DB; mais on log en console pour debug
        print(f"[WARN] Échec log_prediction: {e}")
# -----------------------
# FastAPI
# -----------------------
app = FastAPI(
    title="Attrition API",
    description="Prédiction de probabilité de démission (attrition) via un pipeline scikit-learn.",
    version="0.2.0",
)

@app.get("/")
def root():
    # redirige vers la doc OpenAPI
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    src = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else f"hub:{HF_REPO_ID}/{HF_FILENAME}"
    return {
        "status": "ok",
        "model_source": src,
        "threshold": THRESHOLD,
        "db_connected": bool(db_engine)
    }

@app.post("/predict_one")
def predict_one(emp: EmployeeInput):
    """
    Reçoit un objet EmployeeInput et retourne {threshold, proba, pred}.

    """
    try:
        X = pd.DataFrame([emp.model_dump()])
        proba = float(pipe.predict_proba(X)[:, 1][0])
        pred = int(proba >= THRESHOLD)

        # logging best-effort
        log_prediction(
            source="api",
            inputs=emp.model_dump(),
            proba=proba,
            pred=pred,
            threshold=THRESHOLD,
            status="ok",
        )
        return {"threshold": THRESHOLD, "proba": proba, "pred": pred}
    except Exception as e:
        # log l’erreur côté DB si possible
        log_prediction(
            source="api",
            inputs=emp.model_dump(),
            proba=None, pred=None, threshold=THRESHOLD,
            status="error", error_message=str(e)
        )
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {e}")

@app.post("/predict_proba")
def predict_proba(req: PredictRequest):
    """
    Reçoit {"inputs": [EmployeeInput, ...]} et retourne {threshold, probas[], preds[]}.
    
    """
    try:
        rows = [item.model_dump() for item in req.inputs]
        X = pd.DataFrame(rows)
        probas = pipe.predict_proba(X)[:, 1]
        preds  = (probas >= THRESHOLD).astype(int)

        # logging succès
        log_prediction(
            source="api",
            inputs=rows,
            proba=float(probas.mean()) if len(probas) else None,   # exemple: proba moyenne
            pred=int(preds.mean() >= 0.5) if len(preds) else None, # petite synthèse
            threshold=THRESHOLD
        )
        return {
            "threshold": THRESHOLD,
            "probas": [float(p) for p in probas],
            "preds": preds.tolist(),
        }
    except Exception as e:
        log_prediction(
            source="api",
            inputs=[item.model_dump() for item in req.inputs],
            proba=None, pred=None, threshold=THRESHOLD,
            status="error", error_message=str(e)
        )
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {e}")
