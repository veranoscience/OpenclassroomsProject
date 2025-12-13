from pathlib import Path
import os
import uuid
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from huggingface_hub import hf_hub_download

from sqlalchemy import create_engine, text

# --------- Config ----------
THRESHOLD = 0.33

# Modèle : local puis fallback Hub
LOCAL_MODEL = Path("models/model.joblib")
HF_REPO_ID = "veranoscience/attrition-model"
HF_FILENAME = "model.joblib"

# Base de données (local) :
# export DATABASE_URL="postgresql+psycopg2://appuser:appuser@localhost:5432/attrition"
DATABASE_URL = os.environ.get("DATABASE_URL")
db_engine = create_engine(DATABASE_URL, future=True) if DATABASE_URL else None

# --------- Chargement pipeline ----------
def load_pipeline():
    if LOCAL_MODEL.exists():
        return joblib.load(LOCAL_MODEL)
    downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    return joblib.load(downloaded)

pipe = load_pipeline()

# --------- App ----------
app = FastAPI(
    title="Attrition API",
    description="Prédiction de probabilité de démission (attrition) via un pipeline scikit-learn.",
    version="0.1.0",
)

# --------- Schemas Pydantic ----------
from .schemas import PredictRequest, EmployeeInput  

# --------- Utils logging DB ----------
def log_prediction(source: str, payload: dict, proba: float, pred: int, status: str = "ok", error: str | None = None):
    """Insère 1 ligne dans ml.predictions_log si la DB est configurée."""
    if not db_engine:
        return  
    try:
        with db_engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO ml.predictions_log
                      (request_id, source, input_payload, proba, pred, threshold, model_version, status, error_message)
                    VALUES
                      (:rid, :src, :payload::jsonb, :proba, :pred, :thr, :version, :status, :err)
                """),
                {
                    "rid": str(uuid.uuid4()),
                    "src": source,
                    "payload": json.dumps(payload, ensure_ascii=False),
                    "proba": proba,
                    "pred": pred,
                    "thr": THRESHOLD,
                    "version": "rf_reg@local",   
                    "status": status,
                    "err": error
                }
            )
    except Exception:
        
        pass

# --------- Routes ----------
@app.get("/health")
def health():
    src = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else f"hub:{HF_REPO_ID}/{HF_FILENAME}"
    db = "connected" if db_engine is not None else "not_configured"
    return {"status": "ok", "model_source": src, "threshold": THRESHOLD, "db": db}

@app.post("/predict_proba")
def predict_proba(req: PredictRequest):
    try:
        rows = [item.model_dump() for item in req.inputs]
        X = pd.DataFrame(rows)
        probas = pipe.predict_proba(X)[:, 1]
        preds = (probas >= THRESHOLD).astype(int)

        # log 1 ligne par enregistrement 
        for row, p, y in zip(rows, probas, preds):
            log_prediction(source="api_batch", payload=row, proba=float(p), pred=int(y))

        return {
            "threshold": THRESHOLD,
            "probas": [float(p) for p in probas],
            "preds": preds.tolist(),
        }
    except Exception as e:
        # log erreur (on loggue le lot entier dans 'payload')
        log_prediction(source="api_batch", payload={"batch": [x.model_dump() for x in req.inputs]}, proba=None, pred=None, status="error", error=str(e))
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {e}")

@app.post("/predict_one")
def predict_one(emp: EmployeeInput):
    try:
        X = pd.DataFrame([emp.model_dump()])
        proba = float(pipe.predict_proba(X)[:, 1][0])
        pred = int(proba >= THRESHOLD)

        # log 1 prédiction
        log_prediction(source="api", payload=emp.model_dump(), proba=proba, pred=pred)

        return {"threshold": THRESHOLD, "proba": proba, "pred": pred}
    except Exception as e:
        log_prediction(source="api", payload=emp.model_dump(), proba=None, pred=None, status="error", error=str(e))
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {e}")

@app.get("/")
def root():
    # page racine -> swagger
    return RedirectResponse(url="/docs")