# ZS-Map Outlier Detection & Clustering

This project demonstrates geolocation outlier detection and clustering using the ZS-Map dimensionality reduction technique. It includes tools for generating synthetic geolocation data, performing outlier analysis, and comparing clustering methods.

## Project Structure

- `zsmap/`
  - `synthetic_data.py`: Generates synthetic geolocation data with labeled inliers and outliers.
  - `helper_functions.py`: Utility functions for z-score computation, squaring, summing, percentiles, and plotting.
  - `outlier_detection.py`: Applies the ZS-Map pipeline to synthetic data and visualizes results.
  - `clustering_analysis.py`: Compares ZS-Map quantile-based clustering with K-Means on the Iris dataset.

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn

Install dependencies with:

```sh
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Usage

#### 1. Generate and Analyze Synthetic Data

Run the outlier detection pipeline and visualize results:

```sh
python zsmap/outlier_detection.py
```

This will:
- Generate synthetic latitude/longitude data with outliers.
- Apply the ZS-Map transformation.
- Visualize histograms and a colored scatter plot by percentile group.

#### 2. Clustering Analysis on Iris Dataset

Compare ZS-Map quantile-based clustering with K-Means:

```sh
python zsmap/clustering_analysis.py
```

This will:
- Apply ZS-Map to the Iris dataset.
- Cluster using quantiles and K-Means.
- Print clustering metrics and show comparison plots.

## Key Concepts

- **ZS-Map**: A dimensionality reduction pipeline involving z-score normalization, squaring, summing, square root, and log transformation.
- **Outlier Detection**: Identifies outliers based on percentile thresholds.
- **Clustering Comparison**: Evaluates how ZS-Map quantile clusters align with K-Means clusters.

## File Reference

- [`zsmap/helper_functions.py`](zsmap/helper_functions.py): Utility functions for data transformation and plotting.
- [`zsmap/synthetic_data.py`](zsmap/synthetic_data.py): Synthetic data generation.
- [`zsmap/outlier_detection.py`](zsmap/outlier_detection.py): Main pipeline for outlier detection and visualization.
- [`zsmap/clustering_analysis.py`](zsmap/clustering_analysis.py): Clustering analysis and comparison.

---