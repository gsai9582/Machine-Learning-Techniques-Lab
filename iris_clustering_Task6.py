import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation, Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

print("KMeans Silhouette Score:", silhouette_score(X, kmeans_labels))

# Plot
plt.scatter(X[:,0], X[:,1], c=kmeans_labels)
plt.title("KMeans Clustering")
plt.show()


# ---------------------------
# Affinity Propagation
# ---------------------------
affinity = AffinityPropagation(random_state=42)
affinity_labels = affinity.fit_predict(X)

print("\nEvaluation Metrics for Affinity Propagation")
print("Silhouette Score:", silhouette_score(X, affinity_labels))
print("Davies-Bouldin Index:", davies_bouldin_score(X, affinity_labels))
print("Calinski-Harabasz Index:", calinski_harabasz_score(X, affinity_labels))

plt.scatter(X[:,0], X[:,1], c=affinity_labels)
plt.title("Affinity Propagation Clustering")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()


# ---------------------------
# Birch Clustering
# ---------------------------
birch = Birch(n_clusters=3)
birch_labels = birch.fit_predict(X)

print("\nEvaluation Metrics for Birch Clustering")
print("Silhouette Score:", silhouette_score(X, birch_labels))
print("Davies-Bouldin Index:", davies_bouldin_score(X, birch_labels))
print("Calinski-Harabasz Index:", calinski_harabasz_score(X, birch_labels))

plt.scatter(X[:,0], X[:,1], c=birch_labels)
plt.title("Birch Clustering")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
