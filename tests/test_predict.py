from pathlib import Path

def test_model_artifact_placeholder():
    # à rendre strict après entraînement (vérifier réellement l'artefact)
    assert Path("models").exists()