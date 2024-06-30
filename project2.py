import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data (replace with your own dataset)
# Example using make_blobs
n_samples = 300
n_features = 2
n_clusters = 3
random_state = 42
X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)

# Assuming X is your customer data with purchase behavior features
# Replace X with your actual customer data
# X should be a 2D array where each row is a customer and columns represent features like spending amount, frequency, etc.

# Visualize the generated data (optional)
plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Sample Data for Clustering')
plt.show()

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Visualize the clusters along with centroids (optional for 2D data)
plt.scatter(X[:, 0], X[:, 1], c=labels, marker='o', s=25, cmap='viridis', edgecolor='k')  # Plot points with cluster dependent colors
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, linewidths=3, color='r', label='Centroids')  # Plot centroids
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.legend()
plt.show()

# Example of interpreting the clusters (replace with your analysis)
cluster_df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
cluster_df['Cluster'] = labels
print(cluster_df.head())

# Analyze each cluster (replace with your own interpretation)
for cluster_num in range(n_clusters):
    cluster_data = cluster_df[cluster_df['Cluster'] == cluster_num]
    print(f"\nCluster {cluster_num}:\n{cluster_data.describe()}")

# Replace the sample data generation with your actual customer data and adjust the features accordingly