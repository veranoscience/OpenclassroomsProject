import pandas as pd
from sqlalchemy import create_engine

CSV = "data/processed/df_central_norm.csv"
DB  = "postgresql+psycopg2://appuser:appuser@localhost:5432/attrition"

df = pd.read_csv(CSV)

expected = [
  "age","genre","revenu_mensuel","statut_marital","departement","poste",
  "nombre_experiences_precedentes","annees_dans_le_poste_actuel",
  "note_evaluation_precedente","note_evaluation_actuelle",
  "heure_supplementaires","augementation_salaire_precedente",
  "nombre_participation_pee","nb_formations_suivies",
  "distance_domicile_travail","niveau_education","domaine_etude",
  "frequence_deplacement","annees_depuis_la_derniere_promotion",
  "annes_sous_responsable_actuel",
  "satisfaction_globale","exp_moins_3_years","attrition"
]
missing = [c for c in expected if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes dans CSV: {missing}")

engine = create_engine(DB, future=True)
# mode='append' => on ajoute; si tu veux écraser, passe 'replace' (DANGER).
df.to_sql("employees", engine, schema="hr", if_exists="append", index=False)
print("Import terminé.")
