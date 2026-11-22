import numpy as np
import pandas as pd

# fixed column order to guarantee identical feature alignment across training and inference
COLUNAS = ["doy_sin", "doy_cos", "SST"] + [f"cluster_{c}" for c in range(6)]

# transform raw input dataframe into a numeric feature matrix without mutating the original
def build_X(df):
    d = pd.DataFrame(index=df.index)
    
    # cyclical encoding for day of year
    doy = pd.to_datetime(df["Date"]).dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    
    # cast continuous feature and one-hot encode spatial clusters (0–5)
    d["SST"] = df["SST"].astype(float)
    for c in range(6):
        d[f"cluster_{c}"] = (df["spatial_cluster"] == c).astype(float)
        
    return d[COLUNAS].to_numpy(dtype=float)