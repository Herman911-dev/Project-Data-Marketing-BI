import pandas as pd
 
# 1. CHARGEMENT 
 
df = pd.read_csv('../data/marketing_campaign.csv', sep=';')
 
print(f"Lignes initiales     : {df.shape[0]}")
print(f"Doublons             : {df.duplicated().sum()}")
print(f"Valeurs manquantes :\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")
 
# 2. NETTOYAGE 
 
# Imputation du revenu par la médiane 
df['Income'] = df['Income'].fillna(df['Income'].median())
 
# Suppression des outliers d'âge (nés avant 1940 → incohérents)
df = df[df['Year_Birth'] > 1940]
 
# Suppression des colonnes à variance nulle 
df = df.drop(columns=['Z_CostContact', 'Z_Revenue'], errors='ignore')
 
# Harmonisation du statut marital → dimension binaire Seul / En couple
df['Living_With'] = df['Marital_Status'].replace({
    'Married'  : 'Partner',
    'Together' : 'Partner',
    'Absurd'   : 'Alone',
    'Widow'    : 'Alone',
    'YOLO'     : 'Alone',
    'Divorced' : 'Alone',
    'Single'   : 'Alone',
    'Alone'    : 'Alone',
})
 
# Conversion de la date d'inscription
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='mixed', dayfirst=True)
 
print(f"Nettoyage terminé. Taille finale : {df.shape[0]} lignes.\n")
 
# 3. FEATURE ENGINEERING 
 
# Âge du client
df['Age'] = 2022 - df['Year_Birth']
 
# Panier global : somme des dépenses sur les 6 catégories produits
products = ['MntWines', 'MntFruits', 'MntMeatProducts',
            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['Total_Spent'] = df[products].sum(axis=1)
 
# Nombre total d'achats tous canaux confondus
purchase_channels = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
df['Total_Purchases'] = df[purchase_channels].sum(axis=1)
 
# Nombre total d'enfants au foyer
df['Children_Home'] = df['Kidhome'] + df['Teenhome']
 
# Ancienneté client en jours (référence = date max du dataset)
reference_date = df['Dt_Customer'].max()
df['Seniority_Days'] = (reference_date - df['Dt_Customer']).dt.days
 
# Score de potentiel marketing composite 
campaigns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
             'AcceptedCmp4', 'AcceptedCmp5', 'Response']
df['Total_Promos'] = df[campaigns].sum(axis=1)
 
df['Marketing_Potential_Score'] = (
    (df['Total_Spent']   / df['Total_Spent'].max()   * 70) +
    (df['Total_Promos']  / df['Total_Promos'].max()  * 30)
).round(2)
 
print("Variables créées :", ['Age', 'Total_Spent', 'Total_Purchases',
                              'Children_Home', 'Seniority_Days',
                              'Total_Promos', 'Marketing_Potential_Score'])
 
# 4. EXPORT 
 
df.to_csv('../data/marketing_campaign_cleaned.csv', index=False)
print("\nFichier exporté → ../data/marketing_campaign_cleaned.csv")