import numpy as np
import pandas as pd

# Ordem fixa das colunas para ser igual para o treino e para a previsão
COLUNAS = ["doy_sin", "doy_cos", "SST"] + [f"cluster_{c}" for c in range(6)]

def build_X(df):
    #Transforma a tabela bruta numa matriz de números, sem alterar a original
    d = pd.DataFrame(index=df.index)
    doy = pd.to_datetime(df["Date"]).dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    d["SST"] = df["SST"].astype(float)
    for c in range(6):
        d[f"cluster_{c}"] = (df["spatial_cluster"] == c).astype(float)
    return d[COLUNAS].to_numpy(dtype=float)