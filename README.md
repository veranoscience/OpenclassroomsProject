# Prédiction de l’Attrition des Employés – TechNova Partners

Bienvenue dans ce projet de **classification en Machine Learning** dont l’objectif est d’**analyser et prédire les démissions d’employés (attrition)** au sein de l’ESN _TechNova Partners_, spécialisée dans le conseil en transformation digitale et la vente de solutions SaaS.

Ce dépôt contient l’ensemble du travail réalisé en tant que **Consultant Data Scientist** pour :
- comprendre les **facteurs clés** derrière les démissions,
- construire un **modèle de prédiction de l’attrition**,
- produire des **insights actionnables** pour les équipes RH

## Table des Matières


- [Contexte & objectifs](#-contexte)
- [Jeux de données](#-jeux-de-données)
- [Approche](#-approche)
- [Structure du dépôt](#️-structure-du-dépôt)
- [Mise en place du modèle](#-mise-en-place-du-modèle)
- [API(FastAPI)](#-api-fastapi-api)
- [Base de donées PostgreSQL](#-base-de-données-postgresql)
- [Installation](#️-installation)
- [Utilisation](#️-utilisation)
- [Workflow Git](#workflow-git-branches--commits--tags-workflow)
- [Traçabilité des prédictions](#traçabilité-des-prédiction)
- [Tests & Couverture (CI)](#tests--couverture-ci)
- [Déploiement(CI/CD)](#déploiement-cd-vers-hugging-face-spaces)
- [Déploiment HF Spaces (Docker)](#déploiement-hf-spaces-docker)
- [Livrables](#-livrables)
- [Auteur](#-auteur)

---
<a id="-contexte"></a>

## Contexte & objectifs

**Problème** : anticiper les départs d’employés (attrition) et expliquer les facteurs clés.

**Objectifs**:

- Construire un **pipeline scikit-learn** (prétraitements + modèle RF)
- Exposer une **API ** (FastAPI) avec Swagger/OpenAPI.
- Tracer chaque prédiction (inputs/outputs) dans **PostgreSQL**
- Mettre en place **CI** (tests) et **CD** (déploiement automatique).

---

<a id="-jeux-de-données"></a>

##  Jeux de données

Trois sources principales sont mises à disposition :

1. **SIRH**: poste, département, contrat, âge, ancienneté, salaire, etc.

2. **Évaluations de performance**: notes annuelles, engagement/satisfaction, historiques RH.
3. **Sondage annuel employés**: bien-être, charge, management, équilibre vie pro/perso.  
**Variable cible** (attrition = 1/0)

Ces différentes sources sont **fusionnées et préparées** pour construire un dataset modélisable.

Fichier aligné pour l’inférence : `data/processed/df_central_norm.csv`
(contient le **feature engineering** attendu par le pipeline et les schémas SQL).

---
<a id="-approche"></a>

## Approche

- **Préparation** : encodages catégoriels, normalisation, features (`exp_moins_3_years`, etc.).

- **Modèle** : `RandomForestClassifier` dans un Pipeline scikit-learn.

- **API** : FastAPI + endpoints `predict_one` et `predict_proba`.

- **Traçabilité** : table `ml.predictions_log` (inputs/outputs/threshold).

- **Déploiement** : Docker → Hugging Face Spaces (CD GitHub Actions).

- **Qualité** : Pytest + couverture, règles de branche et PR.

---
<a id="-structure-du-dépôt"></a>

## Structure du dépôt

```text
.
├── src/
│   ├── __init__.py           
│   ├── data_preparation.py.   # chargement & split X/y (données traitées)
│   ├── train_model.py         # entraînement + sauvegarde artefact
│   └── utils.py               # utilitaires (chargement modèle, prédiction unitaire)
├── tests/
│   ├── test_data_preparation.py
│   └── test_predict.py
├── models/                    # artefact modèle 
├── data/
│   ├── raw/                   # fichiers brutsv(privé, ignoré) – .gitkeep 
│   └── processed/             # données traités (visibles)
├── db/
│   ├── attrition_local.session.sql    # schémas hr_staging/hr/ml + indexes + logs
│   ├── create_db.py                   # création table + exemple
│   └── load_csv.py                    # chargement CSV → hr.employees
├── notebooks/
│   ├── 01_analyse_exploratoire.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modelisation.ipynb
├── .github/workflows/
│   ├── ci.yml
│   └── deploy-to-hf-space.yml          
├── main.py                    # entraînement
├── pyproject.toml             # configuration de l'environnement & dépendances
├── requirements.txt.          # exporté depuis uv
├── README.md
├── .gitignore
└── uv.lock
├── Dockerfile                    # verrouillage précis des versions

```
<a id="-mise-en-place-du-modèle"></a>

## Mise en place du modèle

- **Entraînement** : notebooks 01_analyse_exploratoire.ipynb, 02_preprocessing.ipynb et 03_modelisation.ipynb.

- **Artefact modèle** : `model.joblib` (hébergé sur Hugging Face Hub, téléchargé à l’inférence).

- **Chargement côté API** : si `models/model.joblib` est absent, l’API télécharge depuis `veranoscience/attrition-model`.

- **Seuil** : `THRESHOLD=0.33` → `pred` = (proba >= 0.33)`.

---

<a id="-api-fastapi"></a>

## API (FastAPI)

- **Doc Swagger** : `/docs`

- **Endpoints principaux** :

`GET /health` → statut, source du modèle (local ou hub), threshold.

`POST /predict_one` → prédiction unitaire (retourne proba et pred).

`POST /predict_proba` → prédictions batch (retourne listes probas et preds).

Exemple :

```bash
curl -s -X POST http://127.0.0.1:8000/predict_one \
  -H "Content-Type: application/json" \
  -d '{
    "age":41, "genre":"F", "revenu_mensuel":6000, "statut_marital":"Célibataire",
    "departement":"Consulting", "poste":"Consultant",
    "nombre_experiences_precedentes":6, "annees_dans_le_poste_actuel":2,
    "note_evaluation_precedente":3, "note_evaluation_actuelle":3,
    "heure_supplementaires":0, "augementation_salaire_precedente":12,
    "nombre_participation_pee":1, "nb_formations_suivies":2,
    "distance_domicile_travail":5, "niveau_education":2,
    "annees_depuis_la_derniere_promotion":1, "annes_sous_responsable_actuel":2,
    "satisfaction_globale":3.0, "exp_moins_3_years":0,
    "domaine_etude":"Infra & Cloud", "frequence_deplacement":"Occasionnel"
  }'
# → {"threshold":0.33,"proba":0.1443,"pred":0}

```
---

<a id="-base-de-données-postgresql"></a>

## Base de données PostgreSQL

**Objectif** : assurer une traçabilité complète des échanges API ↔ modèle.

- Schémas :

`hr_staging` : réception brute (employees_raw).

`hr` : table typée finale (employees) + index.

`ml` : table de log des inférences (predictions_log) + index.

- Scripts :

`db/attrition_local.session.sql` → crée schémas/tables/index.

`db/load_csv.py` → charge `data/processed/df_central_norm.csv` dans hr.employees.

`db/create_db.py` → exemple alternatif (table simple predictions + insertion d’une ligne).

Activation du logging DB côté API (local) :

```bash
export DATABASE_URL="postgresql+psycopg2://appuser:appuser@localhost:5432/attrition"
uvicorn src.api.server:app --reload
# Chaque appel /predict_* crée une ligne dans ml.predictions_log

```

---
<a id="-installation"></a>

## Installation locale

**Prérequis**

- Python 3.10+

- git

- [uv](https://github.com/astral-sh/uv)

- PostgreSQL 15 local

(macOS : `brew install postgresql@15`)

## Setup Python

```bash
uv venv && source .venv/bin/activate
uv sync
```

## Démarrer l'API

```bash
uvicorn src.api.server:app --reload
# Swagger: http://127.0.0.1:8000/docs
```

---

<a id="-utilisation"></a>

## Utilisation

1. Lancer les notebooks

Depuis la racine du projet, avec l’environnement activé :

```bash
jupyter notebook
```

Puis ouvrir:

- `notebooks/01_analyse_exploratoire.ipynb` pour l’analyse exploratoire

- `notebooks/02_preprocessing.ipynb` pour le nettoyage & feature engineering

- `notebooks/03_modelisation.ipynb` pour la modélisation et SHAP

Script (entrai&nement rapide)

```bash
python main.py
```
Un artefact est sauvegardé dans `models/model.joblib`

Healthcheck
```bash
curl -s http://127.0.0.1:8000/health | jq
```

Batch

```bash
curl -s -X POST http://127.0.0.1:8000/predict_proba \
  -H "Content-Type: application/json" \
  -d '{"inputs":[{ ... les mêmes champs que /predict_one ... }]}'
```

---
<a id="-workflow"></a>

## Workflow Git (branches / commits / tags)

- Branche principale : `main` (protégée)
- Conventions de branches : `<type><-resume->`
  - Types : `feat`, `fix`, `docs`, `refactor`, `chore`, `test`, `data`
  - Examples :  `docs/mise-a-jour-readme`
- Commits descriptifs: `feat: ...`, `chore: ...`
- Tags de version : `v0.1.0`, `v0.2.0`, ...
  - Créer : `git tag -a v0.1.0 -m "v0.1.0: base"`
  - Pousser : `git push origin v0.1.0`

Résolution de conflits : utiliser l’outil intégré **VS Code**
(Accept Current/Incoming → `git add .` → `git rebase --continue`).

---

<a id="-tracabilité"></a>

## Traçabilité des prédictions

Activation : définir la variable d’env. `DATABASE_URL` avant de lancer l’API.

```bash
export DATABASE_URL="postgresql+psycopg2://appuser:appuser@localhost:5432/attrition"
uvicorn src.api.server:app --reload
```
À chaque appel réussi : insertion dans `ml.predictions_log` (date, payload, proba, classe, seuil, statut).

Requête de contrôle :

```sql
SELECT created_at, proba, pred, threshold, status
FROM ml.predictions_log
ORDER BY created_at DESC
LIMIT 5;
```
---

<a id="-tests--couverture-ci"></a>

### Tests & Couverture (CI)

Lancer localement :

```bash
uv run pytest
# avec couverture:
uv run pytest --cov=src --cov-report=term-missing
```

CI GitHub Actions (`.github/workflows/ci.yml`) :

- installe l’environnement,

- exécute Pytest,

- calcule la couverture,

- bloque la PR si la CI échoue.

---
<a id="-déploiement-cd-vers-hugging-face-spaces"></a>

## Déploiement (CD) vers Hugging Face Spaces

- **Dockerfile** à la racine.

- Workflow `deploy-to-hf-space.yml` :

   - clone le repo du Space,

   - copie `Dockerfile`, `pyproject.toml`, `src/, db/, README.md`, etc.,

   - push vers le Space → rebuild automatique → API en ligne.

Secrets requis (GitHub → Settings → Secrets and variables → Actions) :

- `HF_TOKEN` : token Hugging Face (scope write).

- `HF_SPACE_SLUG` : slug owner/SpaceName (ex. veranoscience/OpenClassroomsProject).

Cycle CD :

1. Crée une branche → commit → PR.

2. Merge vers `main`.

3. L’Action CD se lance et déploie sur le Space.

<a id="-docker"></a>

## Déploiement HF Spaces (Docker)

- Dockerfile à la racine. Le Space utilise `sdk: docker`.

Port par défaut `7860`. L’API est servie par Uvicorn.

Modèle téléchargé au démarrage si absent localement.

---

<a id="-livrables"></a>

## Livrables

Le projet fournit :

1. Dépôt Git structuré : code source, `pyproject.toml`, historique (branches/PR/tags), README.

2. API FastAPI fonctionnelle, documentée (Swagger), déployée (Docker + HF Spaces).

3. Tests Pytest + rapport de couverture (via CI).

4. Base PostgreSQL : schémas SQL, scripts de création/chargement, logs d’inférence.

5. CI/CD : GitHub Actions (CI PR & CD déploiement auto).

---


---

<a id="-auteur"></a>

## Auteur

Projet pédagogique - Ksenia Dautel
