import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt

from visualization_utils import plot_confusion_matrix, plot_clusters, plot_anomaly_scores
from utils import load_pretrained_models, train_models, predict, predict_scores_labels, check_and_prepare_data
from streamlit_anomaly_app.utils import analyze_anomalies


st.title("Anomaly Detection App")

mode = st.radio(
    "Step 1: Choose mode",
    options=["Use Pretrained Model", "Train New Model"],
    index=0,
    help="Use an existing model or upload data to train a new one."
)

data_option = st.radio(
    "Step 2: Choose Dataset",
    options=["Use Sample Dataset", "Upload Your Own Dataset"],
    index=0,
    help="Use a provided sample or upload your own CSV file."
)

df, df_train, df_test, df_train_preprocessed, df_test_preprocessed = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

if data_option == "Use Sample Dataset":
    st.info("Using built-in sample dataset.")
    df = pd.read_csv("data/sample_data.csv")
    df_train, df_test, df_train_preprocessed, df_test_preprocessed = check_and_prepare_data(df, skip_preprocessing=True)
    st.success("Sample dataset loaded and checked successfully.")
else:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df_train, df_test, df_train_preprocessed, df_test_preprocessed = check_and_prepare_data(df, skip_preprocessing=False)
        st.success("Custom dataset uploaded and preprocessed successfully.")


st.header("Data Overview")
st.write(df.head())
st.write(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")


if mode == "Use Pretrained Model":
    st.info("Using pretrained models.")
    try:
        preprocessor, iso_forest, gmm, rf = load_pretrained_models()
        st.write("Pretrained models loaded successfully.")
        st.write("You can now proceed to analyze the data or make predictions.")

    except Exception as e:
        st.error(f"Error loading pretrained models: {e}")
        st.stop()

else:
    st.sidebar.header("Model Hyperparameters")

    st.subheader("Isolation Forest Parameters")
    iso_n_estimators = st.sidebar.slider("Number of Estimators", 20, 500, 50, step=10)
    iso_max_samples = st.sidebar.slider("Max Samples", 0.1, 1.0, 0.8, step=0.1)
    iso_contamination = st.sidebar.slider("Contamination", 0.01, 0.5, 0.1, step=0.01)

    st.subheader("GMM Parameters")
    gmm_n_components = st.sidebar.slider("Number of Clusters", 2, 10, 3, step=1)
    gmm_covariance_type = st.sidebar.selectbox("Covariance Type", ['full', 'tied', 'diag', 'spherical'])

    st.subheader("Random Forest Parameters")
    rf_n_estimators = st.sidebar.slider("Number of Trees", 20, 500, 50, step=10)
    rf_max_depth = st.sidebar.slider("Max Depth", 1, 50, 10, step=1)
    min_samples_split = st.sidebar.slider("Min Samples Split", 1, 50, 2, step=1)
    rf_criterion = st.sidebar.selectbox("Criterion", ['gini', 'entropy', 'log_loss'])


    iso_params = {
        'n_estimators': iso_n_estimators,
        'max_samples': iso_max_samples,
        'contamination': iso_contamination
    }
    gmm_params = {'n_components': gmm_n_components, 'covariance_type': gmm_covariance_type}
    rf_params = {
        'n_estimators': rf_n_estimators,
        'max_depth': rf_max_depth,
        'min_samples_split': min_samples_split,
        'criterion': rf_criterion
    }

    st.info("Training new models. This may take a while...")
    df_train, df_test, gmm, iso_forest, rf = train_models(
        df_train,
        df_test,
        df_train_preprocessed,
        iso_params,
        gmm_params,
        rf_params,
    )
    st.success("New models trained successfully.")


st.header("Step 3: Prediction")

st.info("Making predictions on the test dataset...")
metrics, df_test = predict(df_test, df_test_preprocessed, gmm, rf)

st.write(
    f"F1-score: {metrics['f1']:.2f}"
    f"\nPrecision: {metrics['precision']:.2f}"
    f"\nRecall: {metrics['recall']:.2f}"
    f"\nAccuracy: {metrics['accuracy']:.2f}"
)

df_test = predict_scores_labels(df_test, df_test_preprocessed, iso_forest)
st.write(f"Anomalies count in Test Data: {(df_test['anomaly'] == 1).sum()}")


st.header("Step 4: Anomaly Analysis and Detection")

threshold_anomaly_data_percentage = st.slider(
    "Anomaly Data Percentage Threshold",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=0.1,
    help="Percentage of data that can be anomalous before alerting."
)
threshold_anomaly_score_percentile = st.slider(
    "Anomaly Score Percentile Threshold",
    min_value=0.0,
    max_value=100.0,
    value=95.0,
    step=0.1,
    help="Percentile of anomaly scores that can be considered anomalous."
)

st.write(f"Anomaly Data Percentage Threshold: {threshold_anomaly_data_percentage:.2f}%")
st.write(f"Anomaly Score Percentile Threshold: {threshold_anomaly_score_percentile:.2f}%")


analyzed_results = analyze_anomalies(
    test_df=df_test,
    threshold_anomaly_data_percentage=threshold_anomaly_data_percentage,
    threshold_anomaly_score_percentile=threshold_anomaly_score_percentile,
    )

st.write(f"- Most Anomalous Cluster: {analyzed_results['most_anomalous_cluster']}")
st.write(f"- Test anomaly count:  {analyzed_results['test_anomaly_count']}")
st.write(f"- Threshold score :      {analyzed_results['threshold_score']:.2f}")
st.write(f"- Test anomaly rate: {analyzed_results['test_anomaly_pct']:.2f}%")

if analyzed_results["anomalies_exceeded"]:
    st.warning("ALERT: Anomalous behavior detected in test data!")
else:
    st.write("Test data is within normal anomaly rate range.")


st.subheader("Visualizations")


st.markdown("### GMM Clusters")
fig1 = plt.figure()
plot_clusters(X=df_test_preprocessed, cluster_labels=df_test['cluster'], title="GMM Clusters")
st.pyplot(fig1)

st.markdown("### Confusion Matrix")
fig2 = plt.figure()
plot_confusion_matrix(y_true=df_test['cluster'], y_pred=df_test['predicted_cluster'], title="Confusion Matrix")
st.pyplot(fig2)

st.markdown("### Isolation Forest Anomaly Scores")
fig3 = plt.figure()
plot_anomaly_scores(anomaly_scores=df_test['anomaly_score'], title="Test Set Anomaly Scores")
st.pyplot(fig3)
