-- === Schémas ===
CREATE SCHEMA IF NOT EXISTS hr_staging;
CREATE SCHEMA IF NOT EXISTS hr;
CREATE SCHEMA IF NOT EXISTS ml;

-- === STAGING : réception brute du CSV (1 ligne = 1 enregistrement brut en JSON line) ===
DROP TABLE IF EXISTS hr_staging.employees_raw;
CREATE TABLE hr_staging.employees_raw (
  row_id      BIGSERIAL PRIMARY KEY,
  raw_line    JSONB NOT NULL,       -- chaque ligne du CSV convertie en JSON (via COPY program/psql \copy depuis CSV)
  loaded_at   TIMESTAMPTZ DEFAULT NOW()
);

-- === TABLE FINALE HR ===
DROP TABLE IF EXISTS hr.employees;
CREATE TABLE hr.employees (
  id                                SERIAL PRIMARY KEY,
  age                               INT NOT NULL CHECK (age BETWEEN 16 AND 100),
  genre                             TEXT NOT NULL,
  revenu_mensuel                    INT NOT NULL CHECK (revenu_mensuel >= 0),
  statut_marital                    TEXT NOT NULL,
  departement                       TEXT NOT NULL,
  poste                             TEXT NOT NULL,
  nombre_experiences_precedentes    INT NOT NULL CHECK (nombre_experiences_precedentes >= 0),
  annees_dans_le_poste_actuel       INT NOT NULL CHECK (annees_dans_le_poste_actuel >= 0),
  note_evaluation_precedente        INT NOT NULL CHECK (note_evaluation_precedente BETWEEN 1 AND 5),
  note_evaluation_actuelle          INT NOT NULL CHECK (note_evaluation_actuelle BETWEEN 1 AND 5),
  heure_supplementaires             INT NOT NULL CHECK (heure_supplementaires IN (0,1)),
  augementation_salaire_precedente  INT NOT NULL CHECK (augementation_salaire_precedente >= 0),
  nombre_participation_pee          INT NOT NULL CHECK (nombre_participation_pee >= 0),
  nb_formations_suivies             INT NOT NULL CHECK (nb_formations_suivies >= 0),
  distance_domicile_travail         INT NOT NULL CHECK (distance_domicile_travail >= 0),
  niveau_education                  INT NOT NULL CHECK (niveau_education BETWEEN 1 AND 5),
  domaine_etude                     TEXT NOT NULL,
  frequence_deplacement             TEXT NOT NULL,
  annees_depuis_la_derniere_promotion INT NOT NULL CHECK (annees_depuis_la_derniere_promotion >= 0),
  annes_sous_responsable_actuel     INT NOT NULL CHECK (annes_sous_responsable_actuel >= 0),
  satisfaction_globale              NUMERIC(5,2) NOT NULL,
  exp_moins_3_years                 SMALLINT NOT NULL CHECK (exp_moins_3_years IN (0,1)),
  attrition                         SMALLINT NOT NULL CHECK (attrition IN (0,1)),
  inserted_at                       TIMESTAMPTZ DEFAULT NOW()
);

-- Index utiles (ex: filtres et agrégations fréquents)
CREATE INDEX IF NOT EXISTS idx_hr_emp_departement ON hr.employees(departement);
CREATE INDEX IF NOT EXISTS idx_hr_emp_attrition   ON hr.employees(attrition);
CREATE INDEX IF NOT EXISTS idx_hr_emp_poste       ON hr.employees(poste);

-- === LOGS D'INFERENCE ML ===
DROP TABLE IF EXISTS ml.predictions_log;
CREATE TABLE ml.predictions_log (
  request_id     UUID PRIMARY KEY,          -- pour tracer chaque appel
  created_at     TIMESTAMPTZ DEFAULT NOW(),
  source         TEXT NOT NULL DEFAULT 'api',  -- api, batch, test...
  input_payload  JSONB NOT NULL,            -- features bruts reçus par l'API
  proba          NUMERIC(10,6),             -- probabilité (nullable si erreur)
  pred           SMALLINT,                  -- classe
  threshold      NUMERIC(6,4),
  model_version  TEXT,                      -- ex: "rf_reg@2025-12-10"
  status         TEXT NOT NULL DEFAULT 'ok',  -- ok / error
  error_message  TEXT                       -- stack / message en cas d'erreur
);

CREATE INDEX IF NOT EXISTS idx_ml_pred_created_at ON ml.predictions_log(created_at);
CREATE INDEX IF NOT EXISTS idx_ml_pred_status     ON ml.predictions_log(status);
