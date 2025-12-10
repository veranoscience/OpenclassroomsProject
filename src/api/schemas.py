from typing import List
from pydantic import BaseModel

class EmployeeInput(BaseModel):
    age: int
    genre: str
    revenu_mensuel: int
    statut_marital: str
    departement: str
    poste: str
    nombre_experiences_precedentes: int
    annees_dans_le_poste_actuel: int
    note_evaluation_precedente: int
    note_evaluation_actuelle: int
    heure_supplementaires: int
    augementation_salaire_precedente: int
    nombre_participation_pee: int
    nb_formations_suivies: int
    distance_domicile_travail: int
    niveau_education: int
    annees_depuis_la_derniere_promotion: int
    annes_sous_responsable_actuel: int
    satisfaction_globale: float
    exp_moins_3_years: int
    domaine_etude: str
    frequence_deplacement: str

class PredictRequest(BaseModel):
    inputs: List[EmployeeInput]