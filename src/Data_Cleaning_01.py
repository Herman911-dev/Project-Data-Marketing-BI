"""Module de nettoyage et de préparation des données marketing."""

import pandas as pd

# Constantes globales 
INPUT_PATH = "../data/marketing_campaign.csv"
OUTPUT_PATH = "../data/marketing_campaign_cleaned.csv"


def clean_marketing_data(df):
    """Nettoie le DataFrame brut des campagnes marketing.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données brutes.

    Returns:
        pd.DataFrame: Le DataFrame nettoyé.
    """
    # Imputation du revenu par la médiane 
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # Suppression des outliers d'âge
    df = df[df["Year_Birth"] > 1940]

    # Suppression des colonnes à variance nulle
    df = df.drop(columns=["Z_CostContact", "Z_Revenue"], errors="ignore")

    # Harmonisation du statut marital
    marital_mapping = {
        "Married": "Partner",
        "Together": "Partner",
        "Absurd": "Alone",
        "Widow": "Alone",
        "YOLO": "Alone",
        "Divorced": "Alone",
        "Single": "Alone",
        "Alone": "Alone",
    }
    df["Living_With"] = df["Marital_Status"].replace(marital_mapping)

    # Conversion de la date d'inscription 
    df["Dt_Customer"] = pd.to_datetime(
        df["Dt_Customer"], format="mixed", dayfirst=True
    )

    return df


def engineer_features(df):
    """Génère de nouvelles variables (features) pour l'analyse.

    Args:
        df (pd.DataFrame): Le DataFrame nettoyé.

    Returns:
        pd.DataFrame: Le DataFrame enrichi avec les nouvelles variables.
    """
    df["Age"] = 2022 - df["Year_Birth"]

    # Création de variables de façon lisible et aérée
    products = [
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]
    df["Total_Spent"] = df[products].sum(axis=1)

    purchase_channels = [
        "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"
    ]
    df["Total_Purchases"] = df[purchase_channels].sum(axis=1)

    df["Children_Home"] = df["Kidhome"] + df["Teenhome"]

    reference_date = df["Dt_Customer"].max()
    df["Seniority_Days"] = (reference_date - df["Dt_Customer"]).dt.days

    campaigns = [
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
        "AcceptedCmp4", "AcceptedCmp5", "Response"
    ]
    df["Total_Promos"] = df[campaigns].sum(axis=1)

    # Expression sur plusieurs lignes encapsulée dans des parenthèses
    df["Marketing_Potential_Score"] = (
        (df["Total_Spent"] / df["Total_Spent"].max() * 70) +
        (df["Total_Promos"] / df["Total_Promos"].max() * 30)
    ).round(2)

    return df


# Point d'entrée du script protégé
if __name__ == "__main__":
    raw_data = pd.read_csv(INPUT_PATH, sep=";")
    
    print(f"Lignes initiales : {raw_data.shape[0]}")
    
    cleaned_data = clean_marketing_data(raw_data)
    final_data = engineer_features(cleaned_data)
    
    final_data.to_csv(OUTPUT_PATH, index=False)
    print(f"Fichier exporté → {OUTPUT_PATH}")