This repository contains a project focused on detecting performance degradation in distributed data platforms, specifically in systems utilizing Trino and S3-compatible object storage. 
The project explores unsupervised machine learning methods to identify anomalous behavior in SQL execution metrics and system-level logs.
There is a Demo Streamlit web application that provides an interactive interface for visualizing the results of the anomaly detection models. 
It allows users to explore the data, view model predictions, and analyze anomalies in a user-friendly manner.


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

_streamlit_anomaly_app_
This directory contains the Streamlit application and its dependencies:
* _app.py_: The main Streamlit application file that sets up the user interface and handles interactions;
* _utils.py_: A utility module with helper functions for data processing, training, prediction and other common tasks;
* _visualization_utils.py_: A utility module with helper functions for data visualization;
* _data_: A directory containing sample data files used by the Streamlit app for demonstration purposes.
* _models_: A directory containing pre-trained machine learning models used for anomaly detection;
* _dockerfile_: A Dockerfile to build a Docker image for the Streamlit app, allowing for easy deployment and sharing of the application;
* _requirements.txt_: A file listing the Python packages required to run the Streamlit app. 

## streamlit_anomaly_app

To build the application, start Docker app and in the repository root run the command in the terminal:
```bash
cd streamlit_anomaly_app
docker build -t streamlit_anomaly_app .
```
To run the application, execute:
```bash
docker run -p 8501:8501 streamlit_anomaly_app
```
Then, open your web browser and navigate to `http://localhost:8501`.


# Key Features
* Focus on unsupervised learning due to limited labeled anomalies;
* Emphasis on interpretable, low-latency solutions suitable for integration with operational monitoring pipelines;
* Uses metrics aggregated from Trino + S3 workload;

# Requirements
* Python 3.10+
* Jupyter Notebook
* scikit-learn
* streamlit

See _requirements.txt_ for full list of libraries for running Jupyter Notebooks.

See _streamlit_anomaly_app/requirements.txt_ for libraries required to run the Streamlit app.


![](https://komarev.com/ghpvc/?username=elena-dsml)
