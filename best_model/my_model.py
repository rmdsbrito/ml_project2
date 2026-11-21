import pickle
import numpy as np
from utils import build_X

def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict(model, day_and_lag_features_df):
    X = build_X(day_and_lag_features_df)
    Xs = (X - model["mu"]) / model["sd"]
    Xs = np.nan_to_num(Xs, nan=0.0)
    p = model["model"].predict_proba(Xs)[:, 1]
    return np.clip(p, 0.0, 1.0)