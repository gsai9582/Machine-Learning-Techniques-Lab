# =========================================
# MLT LAB – TASK 3
# PCA & LDA ON IRIS DATASET (SINGLE CODE)
# =========================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ---------- LOAD DATASET ----------
iris = load_iris()
X = iris.data
y = iris.target

# ---------- TRAIN-TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# PCA IMPLEMENTATION
# =========================================
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# PCA Visualization (Training Data)
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i in range(3):
    plt.scatter(X_train_pca[y_train == i, 0],
                X_train_pca[y_train == i, 1],
                label=f'Class {i}',
                color=colors[i])
plt.title('PCA - Training Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# PCA Classification
knn_pca = KNeighborsClassifier(n_neighbors=3)
knn_pca.fit(X_train_pca, y_train)
y_pred_pca = knn_pca.predict(X_test_pca)

print("PCA Accuracy:", accuracy_score(y_test, y_pred_pca))

# =========================================
# LDA IMPLEMENTATION
# =========================================
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# LDA Visualization (Training Data)
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.scatter(X_train_lda[y_train == i, 0],
                X_train_lda[y_train == i, 1],
                label=f'Class {i}',
                color=colors[i])
plt.title('LDA - Training Data')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.legend()
plt.show()

# LDA Classification
knn_lda = KNeighborsClassifier(n_neighbors=3)
knn_lda.fit(X_train_lda, y_train)
y_pred_lda = knn_lda.predict(X_test_lda)

print("LDA Accuracy:", accuracy_score(y_test, y_pred_lda))
