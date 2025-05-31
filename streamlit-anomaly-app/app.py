import streamlit as st
import pandas as pd
import time
from utils import load_data, preprocess_data
from models import load_models, predict_realtime

st.set_page_config(page_title="Real-Time Anomaly Detection", layout="wide")
st.title("Real-Time Inference Simulation for Performance Degradation Detection")

st.sidebar.header("Configuration")
data_source = st.sidebar.radio("Choose Dataset", ("Sample Dataset", "Upload Your Own"))

if data_source == "Sample Dataset":
    df = load_data("sample")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a file to continue.")
        st.stop()

n_rows = st.sidebar.slider("Number of rows to simulate", min_value=10, max_value=len(df), value=50)
delay = st.sidebar.slider("Delay between samples (seconds)", 0.0, 3.0, 0.5, step=0.1)
batch_size = st.sidebar.slider("Batch size", min_value=1, max_value=50, value=5, step=1)

preprocessor, gmm, iso_forest, clf = load_models()

n_batches = n_rows // batch_size + (1 if n_rows % batch_size != 0 else 0)
df_selected = df.head(n_rows)
placeholder = st.empty()
log = []

for b in range(n_batches):
    start = b * batch_size
    end = min(start + batch_size, n_rows)
    batch = df_selected.iloc[start:end]

    for i in range(len(batch)):
        sample = batch.iloc[i:i+1]
        x_proc = preprocess_data(sample, preprocessor)
        pred_result = predict_realtime(x_proc, gmm, iso_forest, clf)

        log.append({
            "index": start + i,
            "anomaly": pred_result['anomaly'],
            "cluster": pred_result['cluster'],
            "score": round(pred_result['score'], 3),
            "proba": round(pred_result['proba'], 3)
        })

    log_df = pd.DataFrame(log)
    placeholder.dataframe(log_df, use_container_width=True)
    time.sleep(delay)
