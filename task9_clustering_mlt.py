# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Load Dataset (Iris)
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# 2. Standardize Data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 3. Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(df_scaled)

# 4. Apply Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(df_scaled)

# 5. Evaluation Metrics
kmeans_silhouette = silhouette_score(df_scaled, kmeans_labels)
kmeans_db = davies_bouldin_score(df_scaled, kmeans_labels)

hier_silhouette = silhouette_score(df_scaled, hierarchical_labels)
hier_db = davies_bouldin_score(df_scaled, hierarchical_labels)

# 6. Print Results
print("K-Means Clustering:")
print(f"Silhouette Score: {kmeans_silhouette:.4f}")
print(f"Davies-Bouldin Index: {kmeans_db:.4f}\n")

print("Hierarchical Clustering:")
print(f"Silhouette Score: {hier_silhouette:.4f}")
print(f"Davies-Bouldin Index: {hier_db:.4f}\n")

# 7. Dendrogram
plt.figure(figsize=(10, 5))
linkage_matrix = linkage(df_scaled, method='ward')
dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# 8. Scatter Plots
plt.figure(figsize=(12, 5))

# K-Means Plot
plt.subplot(1, 2, 1)
sns.scatterplot(x=df_scaled[:, 0], y=df_scaled[:, 1],
                hue=kmeans_labels, palette='viridis')
plt.title("K-Means Clustering")

# Hierarchical Plot
plt.subplot(1, 2, 2)
sns.scatterplot(x=df_scaled[:, 0], y=df_scaled[:, 1],
                hue=hierarchical_labels, palette='coolwarm')
plt.title("Hierarchical Clustering")

plt.show()
