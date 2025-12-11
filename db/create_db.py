import os
from sqlalchemy import create_engine, text

# 1) URL par défaut si l'env n'est pas défini
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://appuser:appuser@localhost:5432/attrition"
)

engine = create_engine(DATABASE_URL, future=True, echo=False)

DDL = """
CREATE TABLE IF NOT EXISTS predictions (
  id SERIAL PRIMARY KEY,
  created_at TIMESTAMP DEFAULT NOW(),
  age INT,
  genre TEXT,
  revenu_mensuel INT,
  proba_attrition DOUBLE PRECISION,
  pred INT
);
-- index utile pour le tri par date
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
"""

def main():
    # 2) Création la table (idempotent)
    with engine.begin() as conn:
        for stmt in DDL.strip().split(";"):
            if stmt.strip():
                conn.execute(text(stmt))

    # 3) Insérer un exemple (facultatif mais pratique)
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO predictions(age, genre, revenu_mensuel, proba_attrition, pred)
                VALUES(:age, :genre, :rev, :proba, :pred)
            """),
            {"age": 41, "genre": "F", "rev": 6000, "proba": 0.1443, "pred": 0}
        )

    # 4) Vérification
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT id, created_at, age, pred FROM predictions ORDER BY id DESC LIMIT 3"))
        print("Dernières lignes :")
        for r in rows:
            print(dict(r._mapping))

    print(" Table 'predictions' prête.")

if __name__ == "__main__":
    main()