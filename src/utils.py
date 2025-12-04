"""Utilitaires d'inférence : chargement modèle + prédiction unitaire/batch.

- Respecte l'ordre des colonnes appris à l'entraînement (model.meta.json).
- Accepte dicts (depuis une API) ou Series/DataFrame.
"""
from pathlib import Path
import json
from typing import Iterable, Union, Dict, Any, List
import joblib
import pandas as pd

def _load_meta(models_dir: str | Path = "models") -> dict:
    meta_path = Path(models_dir) / "model.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Métadonnées non trouvées: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))

def load_model(models_dir: str | Path = "models"):
    model_path = Path(models_dir) / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    return joblib.load(model_path)

def _to_dataframe(rows: Union[Dict[str, Any], pd.Series, pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    if isinstance(rows, pd.Series):
        return pd.DataFrame([rows.to_dict()])
    if isinstance(rows, dict):
        return pd.DataFrame([rows])
    if isinstance(rows, Iterable):
        return pd.DataFrame(list(rows))
    raise TypeError("Format d'entrée non supporté pour la prédiction.")

def predict_proba(
    rows: Union[Dict[str, Any], pd.Series, pd.DataFrame, List[Dict[str, Any]]],
    models_dir: str | Path = "models"
) -> List[float]:
    """Retourne les probabilités de la classe positive (attrition=1)."""
    meta = _load_meta(models_dir)
    feats = meta["feature_columns"]

    df = _to_dataframe(rows)
    # assure le même ordre/ensemble de colonnes qu'à l'entraînement
    for col in feats:
        if col not in df.columns:
            # si colonne manquante dans l'entrée, on la crée à 0 (au choix)
            df[col] = 0
    # supprime les colonnes inconnues
    df = df[feats]

    model = load_model(models_dir)
    return [float(p) for p in model.predict_proba(df)[:, 1]]

def predict_proba_one(row: Dict[str, Any], models_dir: str | Path = "models") -> float:
    return predict_proba(row, models_dir=models_dir)[0]