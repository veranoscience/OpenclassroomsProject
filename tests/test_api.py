from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_one_minimal():
    sample = {
        "age": 41,
        "genre": "F",
        "revenu_mensuel": 6000,
        "statut_marital": "Célibataire",
        "departement": "Consulting",
        "poste": "Consultant",
        "nombre_experiences_precedentes": 6,
        "annees_dans_le_poste_actuel": 2,
        "note_evaluation_precedente": 3,
        "note_evaluation_actuelle": 3,
        "heure_supplementaires": 0,
        "augementation_salaire_precedente": 12,
        "nombre_participation_pee": 1,
        "nb_formations_suivies": 2,
        "distance_domicile_travail": 5,
        "niveau_education": 2,
        "annees_depuis_la_derniere_promotion": 1,
        "annes_sous_responsable_actuel": 2,
        "satisfaction_globale": 3.0,
        "exp_moins_3_years": 0,
        "domaine_etude": "Infra & Cloud",
        "frequence_deplacement": "Occasionnel"
    }
    r = client.post("/predict_one", json=sample)
    assert r.status_code == 200
    body = r.json()
    assert "proba" in body and "pred" in body and "threshold" in body
    assert 0.0 <= body["proba"] <= 1.0
    assert body["pred"] in (0, 1)
