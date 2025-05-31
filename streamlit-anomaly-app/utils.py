import pandas as pd
from sklearn.pipeline import Pipeline

def load_data(source="sample"):
    if source == "sample":
        return pd.read_csv("data/sample_dataset.csv")
    else:
        raise NotImplementedError

def preprocess_data(df, pipeline):
    return pipeline.transform(df.drop(columns=["query_ts"], errors="ignore"))
