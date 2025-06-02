import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA


def check_and_prepare_data(df, skip_preprocessing=False, test_size=0.2):
    if df is None or df.empty:
        error_message = "DataFrame is empty or not provided. Please upload a valid CSV file."
        st.error(error_message)
        st.stop()

    if not isinstance(df, pd.DataFrame):
        error_message = "Data is not in DataFrame format. Please upload a valid CSV file."
        st.error(error_message)
        st.stop()

    if 'query_ts' in df.columns:
        df.drop(columns=['query_ts'], inplace=True, errors="ignore")

    df.dropna().reset_index(drop=True, inplace=True)
    df_train = df[:int((1 - test_size) * len(df))]
    df_test = df[int((1 - test_size) * len(df)):]

    if skip_preprocessing:
        return pd.DataFrame(), pd.DataFrame(), df_train, df_test

    preprocessor = Pipeline([
        ('robust_scaler', RobustScaler()),
        ('minmax_scaler', MinMaxScaler()),
        ('pca', PCA(n_components=0.95, random_state=42)),
    ])
    df_train_preprocessed = preprocessor.fit_transform(df_train)
    df_test_preprocessed = preprocessor.transform(df_test)

    return df_train, df_test, df_train_preprocessed, df_test_preprocessed


def load_pretrained_models():
    preprocessor = joblib.load("models/preprocessing_stage1.joblib")
    gmm = joblib.load("models/gmm_model.joblib")
    rf = joblib.load("models/random_forest.joblib")
    iso_forest = joblib.load("models/isolation_forest.joblib")

    return preprocessor, iso_forest, gmm, rf


def train_models(
        df_train,
        df_train_preprocessed,
        iso_params,
        gmm_params,
        rf_params,
        random_state=42,
):
    gmm = GaussianMixture(**gmm_params, random_state=random_state)
    gmm.fit(df_train_preprocessed)
    df_train['cluster'] = gmm.predict(df_train_preprocessed)

    rf = RandomForestClassifier(**rf_params, random_state=random_state)
    rf.fit(df_train_preprocessed, df_train['cluster'])
    df_train['predicted_cluster'] = rf.predict(df_train_preprocessed)

    iso_forest = IsolationForest(**iso_params, random_state=random_state)
    iso_forest.fit(df_train_preprocessed)

    df_train['anomaly'] = iso_forest.predict(df_train_preprocessed)
    df_train['anomaly_score'] = -iso_forest.decision_function(df_train_preprocessed)

    print(f"Train anomalies: {(df_train['anomaly'] == -1).sum()}")

    return df_train, gmm, iso_forest, rf


def predict_clusters(df_test, df_test_preprocessed, gmm, rf):
    y_test = gmm.predict(df_test_preprocessed)
    y_pred = rf.predict(df_test_preprocessed)

    df_test['cluster'] = y_test
    df_test['predicted_cluster'] = y_pred
    predictions = pd.concat([pd.Series(y_test), pd.Series(y_pred)], axis=1)
    predictions.columns = ['cluster', 'predicted_cluster']

    metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
    'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics, df_test, predictions


def predict_scores_labels(df_test, df_test_preprocessed, iso_forest, df_train=None):
    df_test = df_test.copy()
    anomaly_labels = pd.Series(iso_forest.predict(df_test_preprocessed)).replace({1: 0, -1: 1})
    anomaly_scores = -iso_forest.decision_function(df_test_preprocessed)

    df_test['anomaly'] = anomaly_labels
    df_test['anomaly_score'] = anomaly_scores

    if df_train is not None:
        df_train = df_train.copy()
        df_train['anomaly'] = pd.Series(iso_forest.predict(df_test_preprocessed)).replace({1: 0, -1: 1})
        df_train['anomaly_score'] = -iso_forest.decision_function(df_test_preprocessed)

    return df_train, df_test, anomaly_labels, anomaly_scores


def analyze_anomalies(
        predictions,
        threshold_anomaly_data_percentage=5.0,
        threshold_anomaly_score_percentile=95,
):
    cluster_anomaly_rates = predictions.groupby('predicted_cluster')['anomaly'].apply(lambda x: (x == -1).mean())
    most_anomalous_cluster = cluster_anomaly_rates.idxmax()
    threshold_score = np.percentile(predictions['anomaly_score'], threshold_anomaly_score_percentile)

    test_cluster = predictions[predictions['predicted_cluster'] == most_anomalous_cluster]
    test_anomaly_count = (test_cluster['anomaly_score'] >= threshold_score).sum()
    test_anomaly_pct = (test_anomaly_count / predictions.shape[0]) * 100

    anomalies_exceeded = test_anomaly_pct > threshold_anomaly_data_percentage

    return {
        "anomalies_exceeded": anomalies_exceeded,
        "threshold_score": threshold_score,
        "most_anomalous_cluster": most_anomalous_cluster,
        "test_anomaly_pct": test_anomaly_pct if test_anomaly_pct else None,
        "test_anomaly_count": test_anomaly_count if test_anomaly_count else None,
    }
