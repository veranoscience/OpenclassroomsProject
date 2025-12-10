from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from .schemas import PredictRequest, EmployeeInput  # you created these

# Seuil de décision 
THRESHOLD = 0.33

# --- Charge le pipeline (préproc + RF) ---
MODEL_PATH = Path("models/model.joblib")
if not MODEL_PATH.exists():
    raise RuntimeError(f"Modèle introuvable: {MODEL_PATH}. Entraîne et sauvegarde d'abord.")

pipe = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Attrition API",
    description="Prédiction de probabilité de démission (attrition) via un pipeline scikit-learn.",
    version="0.1.0",
)

@app.get("/health")
def health():
    return {"status": "ok", "model": str(MODEL_PATH), "threshold": THRESHOLD}

# --- Endpoint batch: reçoit une liste d'EmployeeInput ---
@app.post("/predict_proba")
def predict_proba(req: PredictRequest):
    """
    Reçoit:
    {
      "inputs": [
        {...},  # EmployeeInput
        {...}
      ]
    }
    Retourne probas et classes (selon THRESHOLD) pour chaque ligne.
    """
    try:
        # Convertit la liste de Pydantic en DataFrame
        rows = [item.model_dump() for item in req.inputs]  # .dict() si pydantic v1
        X = pd.DataFrame(rows)

        # Probabilités (classe positive attrition=1)
        probas = pipe.predict_proba(X)[:, 1]
        # Classes selon le seuil fixe
        preds = (probas >= THRESHOLD).astype(int)

        return {
            "threshold": THRESHOLD,
            "probas": [float(p) for p in probas],
            "preds": preds.tolist()
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