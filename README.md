# Marketing Strategy & Customer Scoring - 2026

##  Objectif du Projet
Ce projet vise à optimiser la stratégie marketing d'une entreprise en utilisant la Data Science et la Business Intelligence. L'objectif est de segmenter la base clients et de prédire le potentiel de conversion pour cibler les campagnes publicitaires de manière plus efficace.

---

##  Stack Technique
* **Analyse & Machine Learning :** Python (Pandas, Scikit-Learn).
* **Algorithmes :** K-Means (Clustering) et Régression Logistique (Scoring).
* **Visualisation :** Power BI Desktop.
* **Gestion de version :** Git / GitHub.

---

##  Étapes Clés
1. **Nettoyage des données :** Traitement des valeurs manquantes et ingénierie des variables (Feature Engineering).
2. **Segmentation Clients (Clustering) :** Identification de 4 profils types (Épicuriens VIP, Jeunes Prudents, Seniors Budgétaires, Petits Consommateurs).
3. **Modèle de Scoring :** Création d'un algorithme prédisant la probabilité de réponse à une campagne marketing.
4. **Dashboard Décisionnel :** Création d'un outil interactif sous Power BI pour piloter la stratégie.

---

##  Résultats du Dashboard
Le dashboard interactif permet de filtrer les données par segment et d'analyser :
* **Revenus globaux :** 1,35M€ de CA analysé.
* **Mix Produit :** Domination du secteur Vin et Viande.
* **Potentiel IA :** Score de conversion moyen de 14,93% (montant à +23% pour les segments VIP).

---

## Robustesse, Limites & Risques
* **Gouvernance :** Le modèle de scoring montre une précision de 85% sur le jeu de test. 
* **Risques :** L'ancienneté des données (comportements pré-2024) peut biaiser les prédictions face à l'inflation actuelle.
* **Amélioration :** Une industrialisation via une API (FastAPI) permettrait de scorer les nouveaux clients en temps réel.

---

## Recommandations Business Finales
1. **Focus VIP :** Allouer 60% du budget marketing sur le segment "Épicuriens VIP" avec des offres exclusives sur les vins premium.
2. **Relance Web :** Améliorer l'UX du site pour le segment "Jeunes Prudents" afin de transformer leur navigation en achat réel.
3. **Cross-Selling :** Développer des bundles "Vin & Viande" pour augmenter le panier moyen des "Seniors Budgétaires".

---

##  Structure du Dépôt
* `/data` : Jeu de données marketing final.
* `/notebooks` : Analyse exploratoire et modèle de Machine Learning.
* `/powerbi` : Fichier `.pbix` et exports PDF des analyses par segment.
* `/src` : Script Python Automatisé.

