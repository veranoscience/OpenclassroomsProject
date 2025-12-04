import pandas as pd
from src.data_preparation import load_processed, split_xy

def test_load_processed_ok():
    df = load_processed()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert "attrition" in df.columns

def test_split_xy_shapes():
    df = load_processed()
    X_train, X_test, y_train, y_test = split_xy(df, target="attrition", test_size=0.2, seed=42)
    assert len(X_train) > 0 and len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)