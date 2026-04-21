"""Module de segmentation (Clustering) et prédiction marketing."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Définition des constantes en haut de fichier
CLEANED_DATA_PATH = "../data/marketing_campaign_cleaned.csv"
FINAL_DATA_PATH = "../data/Marketing_Data_Final.csv"

MODEL_FEATURES = [
    "Income", "Total_Spent", "Age",
    "Children_Home", "Total_Purchases", "Seniority_Days"
]

CLUSTER_NAMES = {
    0: "Seniors Budgétaires",
    1: "Épicuriens VIP",
    2: "Jeunes Prudents",
    3: "Petits Consommateurs",
}


def segment_customers(df, features, n_clusters=4):
    """Applique l'algorithme K-Means pour segmenter les clients.

    Args:
        df (pd.DataFrame): Les données contenant les clients.
        features (list): Liste des colonnes à utiliser pour le clustering.
        n_clusters (int): Le nombre de segments souhaités (défaut 4).

    Returns:
        pd.DataFrame: Le DataFrame enrichi avec les labels de clusters.
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[features])

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        random_state=42,
        n_init=10
    )
    df["Cluster"] = kmeans.fit_predict(x_scaled)
    df["Cluster_Name"] = df["Cluster"].map(CLUSTER_NAMES)

    return df


def train_response_model(df, features, target):
    """Entraîne une régression logistique pour prédire la réponse aux campagnes.

    Args:
        df (pd.DataFrame): Les données d'entraînement.
        features (list): Les variables explicatives.
        target (str): Le nom de la variable cible.

    Returns:
        tuple: Le modèle entraîné et le scaler utilisé.
    """
    x = df[features]
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)
    print("\nRapport de classification (jeu de test) :")
    print(classification_report(y_test, y_pred))

    # Retour constant d'un tuple pour éviter les confusions de type
    return model, scaler


if __name__ == "__main__":
    data = pd.read_csv(CLEANED_DATA_PATH)

    # 1. Clustering
    print("Entraînement K-Means avec k=4 ...")
    data = segment_customers(data, MODEL_FEATURES, n_clusters=4)

    # 2. Modélisation Prédictive
    target_column = "Response"
    trained_model, final_scaler = train_response_model(
        data, MODEL_FEATURES, target_column
    )

    # 3. Scoring final
    x_all_scaled = final_scaler.transform(data[MODEL_FEATURES])
    data["Probabilité_Réponse"] = trained_model.predict_proba(
        x_all_scaled
    )[:, 1].round(4)

    # 4. Export
    data.to_csv(FINAL_DATA_PATH, index=False, encoding="utf-8")
    print(f"\nFichier exporté → {FINAL_DATA_PATH}")