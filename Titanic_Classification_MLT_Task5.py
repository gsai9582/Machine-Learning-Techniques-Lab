# ======================================================
# MLT LAB – TASK 5
# CLASSIFICATION USING TITANIC DATASET
# Algorithms: Decision Tree & Random Forest
# ======================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve
)

# ------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------
titanic = sns.load_dataset("titanic")

# ------------------------------------------------------
# 2. Preprocessing
# ------------------------------------------------------
titanic.dropna(subset=['age', 'embarked'], inplace=True)

X = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
X = pd.get_dummies(X, columns=['sex', 'embarked'], drop_first=True)

y = titanic['survived']

# ------------------------------------------------------
# 3. Train-Test Split
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================================
# 4. DECISION TREE CLASSIFIER
# ======================================================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)

print("\n--- Decision Tree Results ---")
print("Accuracy :", accuracy_dt)
print("Precision:", precision_dt)
print("Recall   :", recall_dt)

# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Precision-Recall Curve
p_dt, r_dt, _ = precision_recall_curve(y_test, dt.predict_proba(X_test)[:, 1])
plt.plot(r_dt, p_dt, label="Decision Tree")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Decision Tree")
plt.legend()
plt.show()

# ======================================================
# 5. RANDOM FOREST CLASSIFIER
# ======================================================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)

print("\n--- Random Forest Results ---")
print("Accuracy :", accuracy_rf)
print("Precision:", precision_rf)
print("Recall   :", recall_rf)

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Precision-Recall Curve
p_rf, r_rf, _ = precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1])
plt.plot(r_rf, p_rf, label="Random Forest", color='green')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Random Forest")
plt.legend()
plt.show()

# ======================================================
# 6. Bar Plot Comparison
# ======================================================
metrics = ['Accuracy', 'Precision', 'Recall']
dt_values = [accuracy_dt, precision_dt, recall_dt]
rf_values = [accuracy_rf, precision_rf, recall_rf]

x = range(len(metrics))
plt.bar(x, dt_values, width=0.4, label='Decision Tree')
plt.bar([i + 0.4 for i in x], rf_values, width=0.4, label='Random Forest')
plt.xticks([i + 0.2 for i in x], metrics)
plt.ylabel("Score")
plt.title("Performance Comparison")
plt.legend()
plt.show()
