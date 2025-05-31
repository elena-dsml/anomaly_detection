This repository contains a project focused on detecting performance degradation in distributed data platforms, specifically in systems utilizing Trino and S3-compatible object storage. The project explores unsupervised machine learning methods to identify anomalous behavior in SQL execution metrics and system-level logs.

# Repository Structure
_exploration/data_preparation.ipynb_
This notebook contains the full preprocessing pipeline:
* Loading and combining SQL query logs and system metrics;
* Data cleaning, aggregation, and resampling;
* Dimensionality reduction using PCA;
* Data visualization and exploratory data analysis (EDA);
* Feature Engineering.

_exploration/modeling_and_evaluation.ipynb_
This notebook includes:
* Application of unsupervised clustering techniques (KMeans, DBSCAN, GMM, Birch, Agglomerative Clustering);
* Application of classification models (SVM, Random Forest, Catboost);
* Model comparison using silhouette score, F1-score, and visual interpretation;
* Performance evaluation.

## streamlit-anomaly-app
This directory contains a Demo Streamlit web application that provides an interactive interface for visualizing the results of the anomaly detection models. 
It allows users to explore the data, view model predictions, and analyze anomalies in a user-friendly manner.

To build the application, start Docker app and in the repository root run the command in the terminal:
```bash
cd streamlit-anomaly-app
docker build -t streamlit-anomaly-app .
```
To run the application, execute:
```bash
docker run -p 8501:8501 streamlit-anomaly-app
```
Then, open your web browser and navigate to `http://localhost:8501`.


# Key Features
* Designed for large-scale, production-grade distributed systems;
* Focus on unsupervised learning due to limited labeled anomalies;
* Emphasis on interpretable, low-latency solutions suitable for integration with operational monitoring pipelines;
* Uses real-world metrics aggregated from Trino + S3 workload;

# Requirements
* Python 3.10+
* Jupyter Notebook
* scikit-learn
* streamlit

See _requirements.txt_ for full list of libraries for running Jupyter Notebooks.

See _streamlit-anomaly-app/requirements.txt_ for libraries required to run the Streamlit app.


![](https://komarev.com/ghpvc/?username=elena-dsml)
