from src.data_preparation import load_processed

def test_load_processed_nonempty():
    df = load_processed()
    assert df.shape[0] > 0