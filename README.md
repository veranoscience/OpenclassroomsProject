# Prédiction de l’Attrition des Employés – TechNova Partners

Bienvenue dans ce projet de **classification en Machine Learning** dont l’objectif est d’**analyser et prédire les démissions d’employés (attrition)** au sein de l’ESN _TechNova Partners_, spécialisée dans le conseil en transformation digitale et la vente de solutions SaaS.

Ce dépôt contient l’ensemble du travail réalisé en tant que **Consultant Data Scientist** missionné par le département RH pour :
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
- [⚙️ Installation](#️-installation)
- [▶️ Utilisation](#️-utilisation)
- [🧾 Livrables](#-livrables)
- [Auteur](#-auteur)

---

## Contexte

TechNova Partners constate un **taux de démission plus élevé que d’habitude**.  

Le responsable SIRH, souhaite :
- objectiver les intuitions recueillies lors des entretiens de départ,
- identifier les **causes racines** de l’attrition,
- disposer d’outils pour **anticiper les risques de démission**.

En parallèle, Manager Data Scientist, accompagne la démarche afin de structurer :
- une **analyse exploratoire solide**,
- un **pipeline de modélisation robuste**,
- une **interprétation claire des résultats** à destination des RH.

Ce projet s’inscrit donc dans un **scénario professionnel complet**, allant de la compréhension métier à la restitution de recommandations.

---

## Objectifs

Le projet vise à :

- **Analyser** les profils des employés ayant quitté l’entreprise vs ceux toujours en poste.
-  **Identifier** les facteurs associés à un risque de démission (ancienneté, salaire, satisfaction, performance, etc.).
- **Construire un modèle de classification** capable de prédire la probabilité de départ d’un employé.
-  **Interpréter le modèle** (via SHAP) pour expliquer les prédictions aux décideurs RH.
- **Fournir des livrables clairs** : notebooks, scripts, environnement reproductible et support de présentation.

---

##  Jeux de données

Trois sources principales sont mises à disposition :

1. **Extrait SIRH**  
   - Poste, département, type de contrat  
   - Âge, ancienneté, niveau de salaire  
   - Variables sociodémographiques  

2. **Extrait des évaluations de performance**  
   - Notes annuelles de performance  
   - Indicateurs de satisfaction / engagement  
   - Historique de certaines évaluations RH

3. **Extrait du sondage annuel employés**  
   - Questions de bien-être au travail  
   - Perception de la charge, du management, de l’équilibre vie pro/vie perso  
   - + une **variable cible** indiquant si l’employé a quitté l’entreprise (attrition = 1/0)

Ces différentes sources sont **fusionnées et préparées** pour construire un dataset exploitable pour la modélisation.

---

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

## Structure du dépôt

.
├── data/
│   ├── raw/               # Fichiers bruts : SIRH, performance, sondage
│   └── processed/         # Données nettoyées / fusionnées
├── notebooks/
│   ├── 01_analyse_exploratoire.ipynb       
│   ├── 02_preprocessing.ipynb
│   └── 03_modelisation.ipynb
├── presentation/
│   └── presentation_tecnova_attrition.pdf
├── pyproject.toml         # Configuration de l'environnement & dépendances
└── README.md

Mise en place du modèle

1. Chargement et préparation

- Import des trois extraits (SIRH, performance, sondage),

- Jointure sur l’identifiant employé,

- Construction de la variable cible (attrition).

2. Prétraitement

Gestion des valeurs manquantes,

Encodage des variables catégorielles (One-Hot, Ordinal, …),

Normalisation / standardisation de certaines variables,

Séparation train/test.

3. Modélisation

Plusieurs modèles de classification sont testés :

Régression Logistique

Random Forest

Dummy

Évaluation à l’aide de :

Accuracy

Precision / Recall

F1-score

ROC-AUC

PR AUC

Matrices de confusion et courbes ROC/PR

Le modèle final retenu est celui offrant le meilleur compromis entre performance et interprétabilité pour les RH.

Interprétabilité avec SHAP

L’interprétation du modèle est réalisée avec SHAP :

- Importance globale des variables

Quelles caractéristiques influencent le plus la probabilité de démission ?

- Explication de cas individuels

Pourquoi tel employé est-il jugé “à risque” par le modèle ?

- Support à la décision RH

Mettre en évidence des leviers d’action : ajustement salarial, mobilité interne, charge de travail, reconnaissance, etc.

Ces analyses sont détaillées dans les notebooks de modélisation et illustrées par des graphiques SHAP (summary plots, force plots…).

Installation

Prérequis

Python 3.10+

git

pip 

Étapes d’installation

1. Cloner le dépôt 
git clone https://github.com/veranoscience/OpenclassroomsProject.git
cd OpenclassroomsProject

2. Créer un environnement virtuel
python -m venv .venv

3. Activer l’environnement virtuel

- Sur Windows :

.\.venv\Scripts\activate


- Sur macOS / Linux :

source .venv/bin/activate

Installer les dépendances

pip install .

Utilisation
1. Lancer les notebooks

Depuis la racine du projet, avec l’environnement activé :

jupyter notebook

Puis ouvrir par exemple :

notebooks/01_analyse_exploratoire.ipynb pour l’analyse exploratoire,

notebooks/02_preprocessing.ipynb pour le nettoyage & feature engineering,

notebooks/03_modelisation.ipynb pour la modélisation et SHAP

Livrables

Le projet fournit :

- Un fichier pyproject.toml décrivant :

la version de Python supportée,

les dépendances nécessaires (pandas, scikit-learn, shap, matplotlib, etc.).

- Des notebooks :

Nettoyage & préparation des données,

Analyse exploratoire,

Modélisation & interprétabilité.

- Un support de présentation (PDF) 

Auteur

Kseniia Dautel