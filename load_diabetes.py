import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Datensatz laden
df = pd.read_csv('diabetes.csv')

print("Datensatz erfolgreich geladen!")
print(df.head())  # Zeigt die ersten Zeilen zur Kontrolle

# IQR-basierte Erkennung von Ausreißern
def detect_outliers_iqr(dataframe):
    df_outlier = dataframe.copy()
    for col in df_outlier.select_dtypes(include=[np.number]).columns:  # Nur numerische Spalten
        Q1 = df_outlier[col].quantile(0.25)
        Q3 = df_outlier[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_outlier[col] = df_outlier[col].apply(lambda x: np.nan if x < lower_bound or x > upper_bound else x)
    return df_outlier

# Anwenden der IQR-Methode
df_outliers = detect_outliers_iqr(df)
print("\n🔍 Fehlende Werte nach Ausreißer-Erkennung:")
print(df_outliers.isnull().sum())

# Entfernen unvollständiger Zeilen
df_complete = df_outliers.dropna()
print("\n Neue Dimensionen nach Bereinigung:", df_complete.shape)

# Merkmalsranking mit Chi-Quadrat
X = df_complete.drop(columns=['class'])  # Unabhängige Variablen
y = df_complete['class']  # Zielvariable

# Skalieren für den Chi-Quadrat-Test
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Berechnen des Chi-Quadrat-Scores
chi_scores, p_values = chi2(X_scaled, y)

# Ergebnisse in DataFrame speichern
chi2_results = pd.DataFrame({'Feature': X.columns, 'Chi2 Score': chi_scores, 'P-Value': p_values})
chi2_results = chi2_results.sort_values(by='Chi2 Score', ascending=False)

print("\n Feature-Ranking nach Chi-Quadrat:")
print(chi2_results)

#  Boxplot & Histogramm für wichtigstes und unwichtigstes Feature
important_feature = chi2_results.iloc[0]['Feature']
least_important_feature = chi2_results.iloc[-1]['Feature']

# Boxplots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df_complete[important_feature])
plt.title(f"Boxplot: {important_feature}")

plt.subplot(1, 2, 2)
sns.boxplot(y=df_complete[least_important_feature])
plt.title(f"Boxplot: {least_important_feature}")
plt.show()

# Histogramme
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df_complete[important_feature], kde=True, bins=20)
plt.title(f"Histogramm: {important_feature}")

plt.subplot(1, 2, 2)
sns.histplot(df_complete[least_important_feature], kde=True, bins=20)
plt.title(f"Histogramm: {least_important_feature}")
plt.show()