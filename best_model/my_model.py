import pickle
import numpy as np

from utils import build_X

# load the serialized model bundle (estimator, scaler stats and feature names)
def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

# generate positive class probability predictions from raw input features
def predict(model, day_and_lag_features_df):
    # construct feature matrix using the exact same pipeline as training
    X = build_X(day_and_lag_features_df)
    
    # standardize using stored training statistics and replace NaNs with 0 (training mean)
    Xs = (X - model["mu"]) / model["sd"]
    Xs = np.nan_to_num(Xs, nan=0.0)
    
    # compute positive class probabilities and enforce strict [0, 1] bounds
    p = model["model"].predict_proba(Xs)[:, 1]
    return np.clip(p, 0.0, 1.0)