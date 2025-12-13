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
_db_engine: Engine | None = None

def get_engine() -> Engine | None:
    """Instancie le moteur SQLAlchemy uniquement si DATABASE_URL est présent et valide."""
    global _db_engine
    if _db_engine is not None:
        return _db_engine
    if not DATABASE_URL:
        return None
    try:
        _db_engine = create_engine(DATABASE_URL, future=True)
        # petit ping à froid
        with _db_engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return _db_engine
    except Exception:
        # Ne bloque pas l’API si la DB n’est pas joignable
        _db_engine = None
        return None

def log_prediction(
    engine: Engine | None,
    *,
    source: str,
    payload: Dict[str, Any],
    proba: float | None,
    pred: int | None,
    threshold: float,
    model_version: str,
    status: str = "ok",
    error_message: str | None = None,
) -> None:
    """
    Insère une ligne dans ml.predictions_log si la DB est disponible.
    Ne lève pas d’exception vers l’API (best effort).
    """
    if engine is None:
        return
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO ml.predictions_log
                    (request_id, source, input_payload, proba, pred, threshold, model_version, status, error_message)
                    VALUES (:rid, :src, :payload::jsonb, :proba, :pred, :thr, :ver, :status, :err)
                """),
                {
                    "rid": str(uuid.uuid4()),
                    "src": source,
                    "payload": json.dumps(payload, ensure_ascii=False),
                    "proba": proba,
                    "pred": pred,
                    "thr": threshold,
                    "ver": "rf_reg@v1",
                    "status": status,
                    "err": error_message,
                },
            )
    except Exception:
        # on n’échoue pas la requête API à cause du logging
        pass

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
    db_mode = "enabled" if get_engine() is not None else "disabled"
    return {
        "status": "ok",
        "model_source": src,
        "threshold": THRESHOLD,
        "database": db_mode,
    }

@app.post("/predict_one")
def predict_one(emp: EmployeeInput):
    """
    Reçoit un objet EmployeeInput et retourne {threshold, proba, pred}.
    Log en DB si DATABASE_URL est défini et la table ml.predictions_log existe.
    """
    engine = get_engine()
    try:
        X = pd.DataFrame([emp.model_dump()])
        proba = float(pipe.predict_proba(X)[:, 1][0])
        pred = int(proba >= THRESHOLD)

        # logging best-effort
        log_prediction(
            engine,
            source="api",
            payload=emp.model_dump(),
            proba=proba,
            pred=pred,
            threshold=THRESHOLD,
            model_version="rf_reg@v1",
            status="ok",
        )
        return {"threshold": THRESHOLD, "proba": proba, "pred": pred}
    except Exception as e:
        # log l’erreur côté DB si possible
        log_prediction(
            engine,
            source="api",
            payload=emp.model_dump(),
            proba=None,
            pred=None,
            threshold=THRESHOLD,
            model_version="rf_reg@v1",
            status="error",
            error_message=str(e),
        )
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {e}")

@app.post("/predict_proba")
def predict_proba(req: PredictRequest):
    """
    Reçoit {"inputs": [EmployeeInput, ...]} et retourne {threshold, probas[], preds[]}.
    Logge 1 ligne par requête (payload=liste complète).
    """
    engine = get_engine()
    rows: List[Dict[str, Any]] = [item.model_dump() for item in req.inputs]
    try:
        X = pd.DataFrame(rows)
        probas = pipe.predict_proba(X)[:, 1]
        preds = (probas >= THRESHOLD).astype(int)

        # logging best-effort (une ligne par lot)
        log_prediction(
            engine,
            source="api-batch",
            payload={"batch": rows},
            proba=float(probas.mean()) if len(probas) else None,  # ex. métrique synthétique
            pred=int(preds.mean() >= 0.5) if len(preds) else None,
            threshold=THRESHOLD,
            model_version="rf_reg@v1",
            status="ok",
        )
        return {
            "threshold": THRESHOLD,
            "probas": [float(p) for p in probas],
            "preds": preds.tolist(),
        }
    except Exception as e:
        log_prediction(
            engine,
            source="api-batch",
            payload={"batch": rows},
            proba=None,
            pred=None,
            threshold=THRESHOLD,
            model_version="rf_reg@v1",
            status="error",
            error_message=str(e),
        )
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {e}")
