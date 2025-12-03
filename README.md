# Prédiction de l’Attrition des Employés – TechNova Partners

Bienvenue dans ce projet de **classification en Machine Learning** dont l’objectif est d’**analyser et prédire les démissions d’employés (attrition)** au sein de l’ESN _TechNova Partners_, spécialisée dans le conseil en transformation digitale et la vente de solutions SaaS.

Ce dépôt contient l’ensemble du travail réalisé en tant que **Consultant Data Scientist** pour :
- comprendre les **facteurs clés** derrière les démissions,
- construire un **modèle de prédiction de l’attrition**,
- produire des **insights actionnables** pour les équipes RH

## Table des Matières


- [Contexte](#-contexte)
- [Objectifs](#-objectifs)
- [Jeux de données](#-jeux-de-données)
- [Approche](#-approche)
- [Structure du dépôt](#️-structure-du-dépôt)
- [Mise en place du modèle](#-mise-en-place-du-modèle)
- [Interprétabilité avec SHAP](#-interprétabilité-avec-shap)
- [Installation](#️-installation)
- [Utilisation](#️-utilisation)
- [Livrables](#-livrables)
- [Auteur](#-auteur)

---
<a id="-contexte"></a>
## Contexte

TechNova Partners constate un **taux de démission supérieur à la normale**.  

Le responsable SIRH, souhaite :
- **objectiver** les hypothèses issues des entretiens de départ,
- **identifier** les **causes racines** de l’attrition,
- **anticiper** les risques de démission.

Accompagnement avec un **pipeline robuste** de modélisation + **restitution claire** pour les RH.

---

<a id="-objectifs"></a>
## Objectifs

- **Analyser** les profils "démissionnaires vs non-démissionnaires"
-  **Identifier** les facteurs associés au risque de démission (ancienneté, salaire, satisfaction, performance, etc.).
- **Construire un modèle de classification** capable de prédire la probabilité de départ d’un employé.
-  **Interpréter le modèle** (via SHAP) (globale & locale)
- **Fournir des livrables clairs** : notebooks, scripts, environnement reproductible et support de présentation.

---

<a id="-jeux-de-données"></a>
##  Jeux de données

Trois sources principales sont mises à disposition :

1. **SIRH**: poste, département, contrat, âge, ancienneté, salaire, etc.

2. **Évaluations de performance**: notes annuelles, engagement/satisfaction, historiques RH.
3. **Sondage annuel employés**: bien-être, charge, management, équilibre vie pro/perso.  
**Variable cible** (attrition = 1/0)

Ces différentes sources sont **fusionnées et préparées** pour construire un dataset modélisable.

---
<a id="-approche"></a>
## Approche

L’analyse suit les grandes étapes suivantes :

1. **Compréhension métier & des données**
   - Lecture des descriptions,
   - Mapping des variables,
   - Identification de la cible.

2. **Nettoyage & préparation**
   - Gestion des valeurs manquantes,
   - Encodage des variables catégorielles,
   - Transformation / normalisation des variables numériques,
   - Jointure des différentes sources de données.

3. **Analyse Exploratoire (EDA)**
   - Statistiques descriptives générales,
   - Comparaisons _démissionnaires_ vs _non-démissionnaires_,
   - Visualisation des distributions et corrélations,
   - Identification de pistes d’explication à tester dans le modèle.

4. **Modélisation**
   - Séparation train/test,
   - Entraînement de plusieurs modèles de classification (Dummy, Logistic Regression, Random Forest),
   - Recherche d’hyperparamètres,
   - Évaluation via des métriques adaptées (PR AUC, ROC-AUC, AUC, Précision, Rappel, F1-score, Seuil de décision )

5. **Interprétabilité**
   - Utilisation de **SHAP** pour comprendre l’impact des variables,
   - Analyse globale (features les plus importantes),
   - Analyse locale (explication de cas particuliers).

6. **Restitution**
   - Synthèse des résultats pour les RH,
   - Recommandations opérationnelles et pistes d’actions.

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
├── notebooks/
│   ├── 01_analyse_exploratoire.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modelisation.ipynb
├── reports           
├── main.py                    # entraînement
├── pyproject.toml             # configuration de l'environnement & dépendances
├── requirements.txt.          # exporté depuis uv
├── README.md
├── .gitignore
└── uv.lock                    # verrouillage précis des versions

```
<a id="-mise-en-place-du-modèle"></a>

## Mise en place du modèle

1. **Chargement et préparation**

- Import des trois extraits (SIRH, performance, sondage),

- Jointure sur l’identifiant employé,

- Construction de la variable cible (attrition).

2. **Prétraitement**

- Gestion des valeurs manquantes,

- Encodage des variables catégorielles (One-Hot, Ordinal, …),

- Normalisation / standardisation de certaines variables,

- Séparation train/test.

3. **Modélisation**

Plusieurs modèles de classification sont testés :

- Régression Logistique

- Random Forest

- Dummy

**Évaluation à l’aide de :**

- Accuracy

- Precision / Recall

- F1-score

- ROC-AUC

- PR AUC

- Matrices de confusion et courbes ROC/PR

Le modèle final retenu est celui offrant **le meilleur compromis entre performance et interprétabilité** pour les RH.

---
<a id="-interprétabilité-avec-shap"></a>
## Interprétabilité avec SHAP

- Importance globale des variables: Quelles caractéristiques influencent le plus la probabilité de démission ? (summary plot)

- Explication de cas individuels: Pourquoi tel employé est-il jugé “à risque” par le modèle ? (force plot)

- Aide à la décision RH: leviers d’action (ajustement salarial, mobilité interne, charge de travail, reconnaissance, etc.)

---
<a id="-installation"></a>
## Installation

**Prérequis**

- Python 3.10+

- git

- [uv](https://github.com/astral-sh/uv)

## Étapes d’installation

1. **Cloner le dépôt**

```bash
git clone https://github.com/veranoscience/OpenclassroomsProject.git

cd OpenclassroomsProject
```

2. **Créer un environnement virtuel**

```bash
uv venv && source .venv/bin/activate
```

3. **Installer les dépendances**

```bash
uv sync
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

---

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

## Authentification & Sécurité

Aucun secret n’est committé
**Compte GitHub** : 2FA activée, Secret Scanning & Dependabot activés.

---

<a id="-livrables"></a>
## Livrables

Le projet fournit :

- **Code source** (notebooks + src/ + main.py)

- **Environnement reproductible** : `pyproject.toml` (uv), `uv.lock`, `requirements.txt` exporté.

- **README** complet (installation, utilisation, sécurité, workflow)

- **Versioning** : historique de commits clair, branches dédiées, tags (ex. v0.1.0).

- **Présentation** : `reports/` (PDF)

---

## Versioning / Changelog

- Version courante : voir tags Git.

- `CHANGELOG.md` pour tracer les évolutions :

v0.1.0 — structure, dépendances, notebooks, entraînement minimal, SHAP.

---

<a id="-auteur"></a>
## Auteur

Kseniia Dautel
