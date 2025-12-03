"""Chargement & split des données traitées."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def load_processed(path: str | Path = "data/processed/df_central_encode.csv") -> pd.DataFrame:
    return pd.read_csv(Path(path))

def split_xy(df: pd.DataFrame, target: str = "attrition", test_size: float = 0.2, seed: int = 42):
     X = df.drop(columns=[target])
     y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
src/train_model.py