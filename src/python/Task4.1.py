import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Datensatz laden
df = pd.read_csv('data/heartdata.csv')
print("Daten geladen:")
print(df.head())

# Korrelation prüfen
print("\nKorrelationstabelle:")
print(df.corr())

# Pearson-Korrelation zwischen biking & heartdisease
corr, p = pearsonr(df['biking'], df['heartdisease'])
print(f"\nPearson-Korrelation (biking vs. heartdisease): {corr:.4f} | p-Wert: {p:.4f}")

# Histogramme
df.hist(figsize=(10, 5))
plt.suptitle("Histogramme der Merkmale", fontsize=14)
plt.tight_layout()
plt.show()

# Pairplot (Streudiagramme)
sns.pairplot(df)
plt.suptitle("Beziehungen zwischen Merkmalen", y=1.02)
plt.show()

# Merkmale & Ziel definieren
X = df[['biking', 'smoking']]  # Features
y = df['heartdisease']         # Zielvariable (kontinuierlich!)

# Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell trainieren
model = LinearRegression()
model.fit(X_train, y_train)

# Modell bewerten
print("\nKoeffizienten:", model.coef_)
print("Achsenabschnitt (Intercept):", model.intercept_)

# Vorhersagen auf Testdaten
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R²-Score auf Testdaten:", r2)

# Vorhersage visualisieren
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Echte Werte (y_test)")
plt.ylabel("Vorhergesagte Werte (y_pred)")
plt.title("Lineare Regression: Vorhersage vs. echte Werte")
plt.grid(True)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonale
plt.show()
