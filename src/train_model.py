"""Entraînement + sauvegarde modèle"""
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from .data_preparation import load_processed, split_xy

def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")

def train_and_save (models_dir: str | Path = "models") -> float:
    df = load_processed()
    X_train, X_test, y_train, y_test = split_xy(df)
    model = build_model()
    model.fit(X_train, y_train)
    f1 = f1_score(y_test, model.predict(X_test))
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(models_dir) / "model.joblib")
    return f1