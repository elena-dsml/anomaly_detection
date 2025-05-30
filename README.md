This repository accompanies a master's thesis focused on detecting performance degradation in distributed data platforms, specifically in systems utilizing Trino and S3-compatible object storage. The project explores unsupervised machine learning methods to identify anomalous behavior in SQL execution metrics and system-level logs.

# Repository Structure
_1_data_preparation.ipynb_
This notebook contains the full preprocessing pipeline:
* Loading and combining SQL query logs and system metrics;
* Data cleaning, aggregation, and resampling;
* Dimensionality reduction using PCA;
* Data visualization and exploratory data analysis (EDA);
* Feature Engineering.

_2_modeling_and_evaluation.ipynb_
This notebook includes:
* Application of unsupervised clustering techniques (KMeans, DBSCAN, GMM, Birch, Agglomerative Clustering);
* Application of classification models (SVM, Random Forest, Catboost);
* Model comparison using silhouette score, F1-score, and visual interpretation;
* Performance evaluation.

# Key Features
* Designed for large-scale, production-grade distributed systems;
* Focus on unsupervised learning due to limited labeled anomalies;
* Emphasis on interpretable, low-latency solutions suitable for integration with operational monitoring pipelines;
* Uses real-world metrics aggregated from Trino + S3 workload;

# Requirements
* Python 3.10+
* Jupyter Notebook
* scikit-learn

See requirements.txt for full list of libraries.
