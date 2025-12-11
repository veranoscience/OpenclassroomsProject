from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download

from .schemas import PredictRequest, EmployeeInput

THRESHOLD = 0.33

LOCAL_MODEL = Path("models/model.joblib")
HF_REPO_ID = "veranoscience/attrition-model"
HF_FILENAME = "model.joblib"

def load_pipeline():
    if LOCAL_MODEL.exists():
        return joblib.load(LOCAL_MODEL)
    downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    return joblib.load(downloaded)

pipe = load_pipeline()

app = FastAPI(
    title="Attrition API",
    description="Prédiction de probabilité de démission (attrition) via un pipeline scikit-learn.",
    version="0.1.0",
)

@app.get("/health")
def health():
    src = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else f"hub:{HF_REPO_ID}/{HF_FILENAME}"
    return {"status": "ok", "model_source": src, "threshold": THRESHOLD}

@app.post("/predict_proba")
def predict_proba(req: PredictRequest):
    try:
        rows = [item.model_dump() for item in req.inputs]
        X = pd.DataFrame(rows)
        probas = pipe.predict_proba(X)[:, 1]
        preds = (probas >= THRESHOLD).astype(int)
        return {
            "threshold": THRESHOLD,
            "probas": [float(p) for p in probas],
            "preds": preds.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {e}")

@app.post("/predict_one")
def predict_one(emp: EmployeeInput):
    try:
        X = pd.DataFrame([emp.model_dump()])
        proba = float(pipe.predict_proba(X)[:, 1][0])
        pred = int(proba >= THRESHOLD)
        return {"threshold": THRESHOLD, "proba": proba, "pred": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {e}")
    
from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/docs")