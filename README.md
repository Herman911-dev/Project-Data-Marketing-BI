# Data Marketing BI — Segmentation & Prédiction Client

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-F7931E.svg?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Data_Processing-Pandas-150458.svg?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626.svg?logo=jupyter&logoColor=white)

**Transformer la donnée brute en stratégie d'acquisition ciblée.** Ce projet d'Intelligence Commerciale (BI) analyse l'historique d'une base de 2 240 clients e-commerce. L'objectif est double : **segmenter** l'audience pour personnaliser les messages, et **prédire** l'appétence aux futures campagnes marketing grâce au Machine Learning afin d'optimiser le budget d'acquisition.

---

## Problématique Business
> *"À qui parler, avec quel message, et dans quel ordre de priorité ?"*

Une entreprise dispose de milliers de lignes de transactions et de retours de campagnes. Plutôt que d'envoyer des offres de manière uniforme (et coûteuse) à toute la base, ce projet fournit un outil d'aide à la décision pour concentrer les efforts sur les clients à fort potentiel.

### 💡 Chiffres Clés & Résultats
| Indicateur | Valeur | Description |
| :--- | :--- | :--- |
| **Volume de données** | `2 240 clients` | Base de données analysée et nettoyée |
| **Segments créés** | `4 profils` | Identifiés par clustering (K-Means) |
| **Score Prédictif** | `86 % Accuracy` | Prédiction de conversion (Régression Logistique) |
| **Création de Valeur** | `Score Marketing` | Nouveau KPI composite créé de 0 à 100 |

---

## Pipeline Analytique & Machine Learning

### Data Processing & Feature Engineering (ETL)
Nettoyage rigoureux (`data_cleaning.py`) pour garantir la fiabilité des modèles :
* **Imputation** des revenus manquants par la médiane.
* **Traitement des Outliers** (suppression des dates de naissance incohérentes < 1940).
* **Création de variables métier (Features) :** `Age`, `Total_Spent`, `Total_Purchases`, `Children_Home`, `Seniority_Days`, et le nombre d'offres acceptées (`Total_Promos`).
* **Innovation:** Création d'un **Marketing Potential Score (0-100)** pondérant le volume de dépenses (70%) et la réactivité historique (30%).

### Segmentation Client (K-Means Clustering)
Utilisation de la méthode du coude (*Elbow Method*) pour déterminer `k=4`.
* **Épicuriens VIP :** Revenu élevé, fort panier, peu d'enfants. *Le moteur de rentabilité.*
* **Seniors Budgétaires :** Clients fidèles mais contraints par des charges familiales.
* **Jeunes Prudents :** Faible revenu, dépenses contenues mais fort potentiel de croissance.
* **Petits Consommateurs :** Peu engagés, très faible réactivité aux campagnes.

### Modélisation Prédictive (Régression Logistique)
Prédiction binaire de la variable `Response` (Le client va-t-il accepter la prochaine offre ?).
* Split stratifié 70% Train / 30% Test avec normalisation `StandardScaler`.
* Génération d'un **score de probabilité (0 à 1)** pour trier la base et prioriser les appels/emails commerciaux.

---

## Stack Technique

* **Langage :** Python
* **Machine Learning :** `scikit-learn` (KMeans, LogisticRegression, StandardScaler)
* **Manipulation de données :** `pandas`, `numpy`
* **Data Visualisation :** `matplotlib`, `seaborn`

---

## Structure du Projet

```text
Project-Data-Marketing-BI/
├── data/
│   ├── marketing_campaign.csv          # Dataset brut source
│   ├── marketing_campaign_cleaned.csv  # Données après pipeline ETL
│   └── Marketing_Data_Final.csv        # Export avec Segments & Scores (Prêt pour la BI)
├── notebooks/
│   └── exploration_nettoyage.ipynb     # Exploratory Data Analysis (EDA) & Modélisation
├── src/
│   ├── data_cleaning.py                # Script ETL (Nettoyage + Feature Engineering)
│   └── model.py                        # Entraînement ML (Clustering + Classification)
├── requirements.txt                    # Dépendances du projet
└── README.md