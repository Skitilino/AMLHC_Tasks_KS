import pandas as pd

# Lade das Dataset
df = pd.read_csv('food.csv')

# Zeige die ersten 5 Zeilen
print(df.head())
from sklearn.preprocessing import StandardScaler

# Entferne die erste Spalte (Ländernamen)
features = df.drop(columns=df.columns[0])

# Skaliere die Daten mit z-Transformation
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Optional: Zeige transformierte Daten (nur die ersten 5 Zeilen)
import numpy as np
np.set_printoptions(precision=2, suppress=True)
print(scaled_features[:5])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

best_k = None
best_score = -1
best_model = None

print("\nSilhouette Scores:")
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, labels)
    print(f"k={k} → Silhouette Score: {score:.3f}")
    
    if score > best_score:
        best_k = k
        best_score = score
        best_model = kmeans

print(f"\nBestes k: {best_k} mit Silhouette Score: {best_score:.3f}")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


from matplotlib.lines import Line2D
import seaborn as sns

# PCA vorbereiten (falls noch nicht geschehen)
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)
labels = best_model.labels_
countries = df[df.columns[0]].values  # Ländernamen

# Farben festlegen
palette = sns.color_palette("Set2", len(set(labels)))

# Plot erstellen
plt.figure(figsize=(12, 8))
for i in range(len(pca_features)):
    plt.scatter(pca_features[i, 0], pca_features[i, 1], 
                color=palette[labels[i]], s=120, edgecolor='black')
    plt.text(pca_features[i, 0] + 0.1, pca_features[i, 1] + 0.1, 
             countries[i], fontsize=9)

# Legende generieren
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
           markerfacecolor=palette[i], markersize=10, markeredgecolor='k')
    for i in range(len(set(labels)))
]

plt.legend(handles=legend_elements, title="Cluster")
plt.title(f'KMeans Clustering (k={best_k}) auf PCA-reduzierten Daten')
plt.xlabel('PCA-Komponente 1 – Hauptunterschiede')
plt.ylabel('PCA-Komponente 2 – zweithäufigste Variation')
plt.grid(True)
plt.tight_layout()
plt.show()



from scipy.cluster.hierarchy import linkage, dendrogram

# Hierarchisches Clustering (Linkage)
linked = linkage(scaled_features, method='ward')

# Dendrogramm plotten
plt.figure(figsize=(12, 6))
dendrogram(linked, labels=df[df.columns[0]].values, orientation='top', distance_sort='descending')
plt.title('Hierarchisches Clustering Dendrogramm')
plt.xlabel('Länder')
plt.ylabel('Abstand')
plt.tight_layout()
plt.show()


import seaborn as sns

# Setze den Ländernamen als Index (für die Achsenbeschriftung)
heatmap_data = pd.DataFrame(scaled_features, 
                            index=df[df.columns[0]], 
                            columns=features.columns)

# Heatmap mit Clustern
sns.clustermap(heatmap_data, cmap='viridis', figsize=(12, 8), method='ward')
plt.show()

from sklearn.cluster import DBSCAN

# DBSCAN Clustering
dbscan = DBSCAN(eps=1.5, min_samples=2)  # eps & min_samples ggf. anpassen
db_labels = dbscan.fit_predict(scaled_features)

# Anzahl Cluster (ohne Rauschen)
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
print(f"\nDBSCAN erkannte {n_clusters} Cluster")

# PCA-Visualisierung der DBSCAN-Ergebnisse
plt.figure(figsize=(8, 6))
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=db_labels, cmap='plasma', s=50)
plt.title('DBSCAN Clustering mit PCA')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.tight_layout()
plt.show()

