import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
 
# 1. CHARGEMENT 
 
df = pd.read_csv('../data/marketing_campaign_cleaned.csv')
 
# 2. VARIABLES DE CLUSTERING 
 
CLUSTER_FEATURES = [
    'Income', 'Total_Spent', 'Age',
    'Children_Home', 'Total_Purchases', 'Seniority_Days'
]
 
X_cluster = df[CLUSTER_FEATURES]
 
# 3. STANDARDISATION 
 
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
 
# 4. MÉTHODE DU COUDE
 
print("Calcul de l'inertie pour k = 1 à 10 ...")
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    km.fit(X_cluster_scaled)
    wcss.append(km.inertia_)
    print(f"  k={k:2d}  →  inertie = {km.inertia_:.2f}")
 
# 5. K-MEANS FINAL (k=4) 
 
print("\nEntraînement K-Means avec k=4 ...")
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)
 
CLUSTER_NAMES = {
    0: 'Seniors Budgétaires',
    1: 'Épicuriens VIP',
    2: 'Jeunes Prudents',
    3: 'Petits Consommateurs',
}
df['Cluster_Name'] = df['Cluster'].map(CLUSTER_NAMES)
 
# Résumé des segments
summary = df.groupby('Cluster_Name')[CLUSTER_FEATURES].mean().round(2)
summary['Count'] = df['Cluster_Name'].value_counts()
print("\nPortrait-robot par segment :")
print(summary.to_string())
 
# 6. KPI PAR SEGMENT 
 
kpi = df.groupby('Cluster_Name').agg(
    Revenu_Moyen        =('Income',                   'mean'),
    Panier_Moyen        =('Total_Spent',              'mean'),
    Taux_Conversion_Pct =('Response',                 lambda x: round(x.mean() * 100, 2)),
    Score_Potentiel     =('Marketing_Potential_Score', 'mean'),
    Nb_Clients          =('ID',                        'count'),
).round(2)
 
print("\nTableau de pilotage KPI :")
print(kpi.to_string())
 
# 7. MODÈLE PRÉDICTIF — RÉGRESSION LOGISTIQUE 
 
MODEL_FEATURES = [
    'Income', 'Total_Spent', 'Age',
    'Children_Home', 'Total_Purchases', 'Seniority_Days'
]
TARGET = 'Response'
 
X = df[MODEL_FEATURES]
y = df[TARGET]
 
# Split 70/30 stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
 
# Normalisation
scaler_model = StandardScaler()
X_train_scaled = scaler_model.fit_transform(X_train)
X_test_scaled  = scaler_model.transform(X_test)
 
# Entraînement
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)
 
# Évaluation
y_pred = model.predict(X_test_scaled)
print("\nRapport de classification (jeu de test) :")
print(classification_report(y_test, y_pred))
 
# 8. SCORING — PROBABILITÉ DE RÉPONSE POUR TOUS LES CLIENTS 
 
X_all_scaled = scaler_model.transform(df[MODEL_FEATURES])
df['Probabilité_Réponse'] = model.predict_proba(X_all_scaled)[:, 1].round(4)
 
print("Top 5 clients les plus appétents :")
print(df[['ID', 'Cluster_Name', 'Probabilité_Réponse']]
      .sort_values('Probabilité_Réponse', ascending=False)
      .head().to_string(index=False))
 
# 9. EXPORT FINAL 
 
df.to_csv('../data/Marketing_Data_Final.csv', index=False, encoding='utf-8')
print("\nFichier exporté → ../data/Marketing_Data_Final.csv")