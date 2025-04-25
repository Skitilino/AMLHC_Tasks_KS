# Task 4.2 - Supervised Learning - Classification

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv('data/diabetes.csv')

# Features und Zielvariable definieren
X = df.drop('class', axis=1)
y = df['class']

# 2. Fit a Generalized Linear Model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 3. Evaluate the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print("Accuracy (auf Gesamtdaten):", accuracy)

# Residuen visualisieren
residuals = y != y_pred
plt.plot(residuals.values, 'o')
plt.title("Residuals (Fehlklassifikationen)")
plt.xlabel("Beispiel-Index")
plt.ylabel("Residual")
plt.show()

# 4. Train/test split (Cross-validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell auf Trainingsdaten fitten
model.fit(X_train, y_train)

# Auf Testdaten evaluieren
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)
