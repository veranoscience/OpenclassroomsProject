"""Entraînement + sauvegarde du modèle final (RandomForest régularisé)."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from .data_preparation import load_processed, split_xy

# Le modèle final choisi 
def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=8,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

def train_and_save(models_dir: str | Path = "models",
                   data_path: str | Path = "data/processed/df_central_encode.csv",
                   target: str = "attrition") -> float:
    df: pd.DataFrame = load_processed(data_path)
    X_train, X_test, y_train, y_test = split_xy(df, target=target, test_size=0.2, seed=42)

    model = build_model()
    model.fit(X_train, y_train)

    f1 = f1_score(y_test, model.predict(X_test))

    # Sauvegardes
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1) modèle
    joblib.dump(model, models_dir / "model.joblib")

    # 2) métadonnées utiles à l’inférence
    meta = {
        "feature_columns": list(X_train.columns),
        "target": target,
        "trained_on": str(data_path),
        "metrics": {"f1_test": float(f1)},
        "model": "RandomForestClassifier",
        "params": {
            "n_estimators": 200,
            "class_weight": "balanced",
            "max_depth": 8,
            "min_samples_leaf": 5,
            "min_samples_split": 10,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": 42,
        },
    }
    (models_dir / "model.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return f1