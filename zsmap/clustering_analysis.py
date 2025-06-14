import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix

# Load the iris dataset from sklearn
iris = load_iris()
X = iris.data
y = iris.target

# Define the ZS-Map dimensionality reduction pipeline
def zs_map_transform(X):
    # Compute z-scores
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    Z = (X - mu) / sigma
    
    # Square the scores
    Z_squared = Z**2
    
    # Reduce dimensions by taking the sum of the squares
    s = np.sum(Z_squared, axis=1)
    
    # Take the square root of the result set
    r = np.sqrt(s)
    
    # Convert log normal dist to normal dist by taking the log of the results
    l = np.log(r)
    
    # Return the reshaped result
    return l.reshape(-1, 1)

# Apply ZS-Map to iris dataset
X_zmap = zs_map_transform(X)

# Create quantile-based clusters
# For iris we need three groups
quantiles = np.percentile(X_zmap, [70, 90])

# Create the quantil clusters using the above
def quantile_cluster(score):
    if score <= quantiles[0]:
        return 0
    elif score <= quantiles[1]:
        return 1
    else:
        return 2

# Set labels
labels_zmap_quantile = np.array([quantile_cluster(s) for s in X_zmap.flatten()])

# Run K-Means clustering 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
labels_kmeans = kmeans.labels_

# Compare clusters using ari and nmi
ari = adjusted_rand_score(labels_kmeans, labels_zmap_quantile)
nmi = normalized_mutual_info_score(labels_kmeans, labels_zmap_quantile)

print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

# Plot result set
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(np.arange(len(X_zmap)), X_zmap.flatten(), c=labels_kmeans, cmap='viridis', s=50)
plt.title('K-Means Clusters (Original Data)')
plt.xlabel('Sample Index')
plt.colorbar(label='Cluster Label')

plt.subplot(1, 2, 2)
plt.scatter(np.arange(len(X_zmap)), X_zmap.flatten(), c=labels_zmap_quantile, cmap='plasma', s=50)
plt.title('Quantile-based ZS-Map Clusters')
plt.xlabel('Sample Index')
plt.colorbar(label='Quantile Cluster')

plt.tight_layout()
plt.show()

# Generate confusion matrix
cm = confusion_matrix(labels_kmeans, labels_zmap_quantile)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('ZS-Map Cluster')
plt.ylabel('K-Means Cluster')
plt.title('Confusion Matrix: K-Means vs ZS-Map Clusters')
plt.show()

