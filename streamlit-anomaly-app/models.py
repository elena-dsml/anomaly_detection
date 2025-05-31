import joblib

def load_models():
    preprocessor = joblib.load("models/preprocessor.joblib")
    gmm = joblib.load("models/gmm.joblib")
    iso_forest = joblib.load("models/isolation_forest.joblib")
    clf = joblib.load("models/classifier.joblib")
    return preprocessor, gmm, iso_forest, clf

def predict_realtime(x, gmm, iso_forest, clf):
    cluster = gmm.predict(x)[0]
    score = -iso_forest.decision_function(x)[0]
    anomaly = int(score > iso_forest.threshold_ if hasattr(iso_forest, 'threshold_') else score > 0.5)
    proba = clf.predict_proba(x)[0][1]
    return {"anomaly": anomaly, "cluster": cluster, "score": score, "proba": proba}
