# COX-2 Inhibitor Classification - Vollständiger Code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, cohen_kappa_score,
    roc_auc_score, roc_curve, classification_report
)

# 1. Daten laden

df = pd.read_csv('data/cox2-2.csv')

# Ziel und Features definieren
X = df.drop(columns=['IC50', 'cox2Class'])
y = df['cox2Class'].map({'Inactive': 0, 'Active': 1})  # Ziel in 0/1 umwandeln

# 2. Trainings-/Test-Split (75 % / 25 %)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Skalierung

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))

# 4. Hyperparameter-Tuning mit GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [50, 100],  # Weniger Bäume für schnellere Ergebnisse
    'max_depth': [5, 10],  # Begrenzte Tiefe für schnellere Berechnung
    'max_features': ['sqrt']  # Einfachere Berechnung
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=10,  # Weniger Kombinationen
    cv=3,  # Weniger Folds
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_scaled, y_train)
best_rf = random_search.best_estimator_


grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    cv=10,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
best_rf = grid_search.best_estimator_

# 5. ROC-Kurve

y_proba = best_rf.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color='orange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-Kurve")
plt.legend(loc="lower right")
plt.show()

# 6. Feature-Importances

importances = best_rf.feature_importances_
indices = importances.argsort()[::-1][:10]
features = X.columns[indices]

plt.figure(figsize=(12, 6))
sns.barplot(y=importances[indices], x=features, palette="Oranges_r")
plt.title("Top 10 Feature-Importances", fontsize=16)
plt.ylabel("Importance", fontsize=14)
plt.xlabel("Feature", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()

# 7. Konfusionsmatrix

y_pred = best_rf.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Oranges", cbar=False, 
            xticklabels=["Inactive", "Active"], yticklabels=["Inactive", "Active"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 8. Klassifikationsbericht

report = classification_report(y_test, y_pred, target_names=["Inactive", "Active"])
print("\nClassification Report:\n", report)
