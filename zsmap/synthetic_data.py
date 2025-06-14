import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_data():
    """
    Helper function used to generate synthetic geolocation data.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Set Parameters
    num_inliers = 1000
    num_outliers = 40

    # Generate Inliers
    inlier_latitudes = np.random.uniform(38.5, 38.75, num_inliers)
    inlier_longitudes = np.random.uniform(-90.4, -90.1, num_inliers)
    inlier_labels = np.zeros(num_inliers)

    # Generate Outliers
    outlier_latitudes = np.random.uniform(38.3, 38.4, num_outliers)
    outlier_longitudes = np.random.uniform(-90.5, -90.0, num_outliers)
    outlier_labels = np.ones(num_outliers)

    # Combine inliers and outliers into one dataset
    latitudes = np.concatenate([inlier_latitudes, outlier_latitudes])
    longitudes = np.concatenate([inlier_longitudes, outlier_longitudes])
    labels = np.concatenate([inlier_labels, outlier_labels])

    # Create DataFrame
    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'is_outlier': labels.astype(int)
    })

    # Visualize the synthetic data
    plt.figure(figsize=(8,6))
    plt.scatter(df[df.is_outlier == 0].longitude, df[df.is_outlier == 0].latitude, c='blue', label='Inliers', alpha=0.6)
    plt.scatter(df[df.is_outlier == 1].longitude, df[df.is_outlier == 1].latitude, c='red', label='Outliers', alpha=0.8)
    plt.title("Synthetic GPS Data")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    return df
