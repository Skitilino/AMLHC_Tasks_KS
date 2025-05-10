import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Load the dataset
df = pd.read_csv('data/BloodBrain-2.csv')

# Daten prüfen
print("Data Shape:", df.shape)
print("Data Info:")
df.info()
print("Erste 5 Zeilen:")
print(df.head())

# Features und Zielvariable trennen
X = df.drop(columns=['logBBB'])
y = df['logBBB']

# Train-Test-Split (75% Training, 25% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Daten skalieren
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modell initialisieren
model = RandomForestRegressor(random_state=42)

# Hyperparameter-Suche (RandomizedSearchCV)
param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
search = RandomizedSearchCV(
    model, param_dist, cv=5, n_jobs=-1, n_iter=20, random_state=42
)
search.fit(X_train_scaled, y_train)

# Beste Parameter und Score ausgeben
print("Beste Parameter:", search.best_params_)
print("Beste R2-Bewertung (CV):", search.best_score_)

# Finales Modell trainieren
best_model = search.best_estimator_
best_model.fit(X_train_scaled, y_train)

# Vorhersage und Evaluation
y_pred = best_model.predict(X_test_scaled)
print("Test MSE:", mean_squared_error(y_test, y_pred))
print("Test R2 Score:", r2_score(y_test, y_pred))

# Top 10 Feature Importances
importances = best_model.feature_importances_
features = X.columns
imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(10)

# Plot
plt.figure(figsize=(12, 6))
plt.bar(imp_df['Feature'], imp_df['Importance'], color='tab:blue')
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Feature Importances')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()
