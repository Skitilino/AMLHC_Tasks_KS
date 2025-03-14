import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1️ Datensatz laden
file_path = "food.csv"  # Falls die Datei in demselben Ordner liegt
food_data = pd.read_csv(file_path)

# Überblick über die Daten
print(food_data.shape)  # Dimensionen des Datensatzes
print(food_data.info())  # Datentypen und fehlende Werte
print(food_data.isnull().sum())  # Anzahl fehlender Werte

# Falls die erste Spalte (Länder) nur Bezeichnungen enthält, als Index setzen
food_data.set_index(food_data.columns[0], inplace=True)

# 2️ Feature Scaling (Z-Transformation)
scaler = StandardScaler()
food_scaled = scaler.fit_transform(food_data)

# 3️ PCA durchführen
pca = PCA(n_components=2)
pca_result = pca.fit_transform(food_scaled)

# PCA-Ergebnisse als DataFrame speichern
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"], index=food_data.index)

# 4️ Score-Plot mit PC1 und PC2
plt.figure(figsize=(10, 6))
plt.scatter(pca_df["PC1"], pca_df["PC2"], color='blue', alpha=0.7)

# Ländernamen als Labels hinzufügen
for country, (x, y) in pca_df.iterrows():
    plt.text(x, y, country, fontsize=9, ha='right')

plt.title("PCA Score Plot: PC1 vs. PC2")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()
