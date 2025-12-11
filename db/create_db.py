import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]
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
"""

if __name__ == "__main__":
    with engine.begin() as conn:
        conn.execute(text(DDL))
    print("Table 'predictions' prête.")