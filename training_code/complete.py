import pandas as pd
import numpy as np
from sklearn import base
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

modelos = {
    "logreg C=0.01": lambda: LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000),
    "logreg C=0.1":  lambda: LogisticRegression(C=0.1,  class_weight="balanced", max_iter=1000),
    "logreg C=1":    lambda: LogisticRegression(C=1,    class_weight="balanced", max_iter=1000),
    "logreg C=10":   lambda: LogisticRegression(C=10,   class_weight="balanced", max_iter=1000),
    "arvore":  lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, class_weight="balanced", random_state=0),
    "rf":      lambda: RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=0),
    "mlp":     lambda: MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-2, max_iter=1000, random_state=0),
}

iteracoes = {
    
}


def ridge_fit(X, y, lam=1.0):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])   # X^T.X + lambda.I = A
    w = np.linalg.solve(A, Xs.T @ (y - y.mean()))  # resolve o sistema A.w = X^T.(y - y.mean()), em func a w
    return mu, sd, w, y.mean()

def ridge_predict(X, mu, sd, w, b):
    return ((X - mu) / sd) @ w + b

def teste_ridge_fit(df, run=True):
    if run:
        # Ciclo para testar as correlações entre variáveis
        comparison = df[df["CHL"].notna()].copy()
        comparison["log_CHL"] = np.log(comparison["CHL"]) 
    
        cols = ["SST", "UI", "doy_sin", "doy_cos"] + [f"cluster_{i}" for i in range(6)] 
        print(comparison[cols + ["log_CHL"]].corr()["log_CHL"].round(2))
        
        # Separação entre datasets de treino, de validação
        tr, va = comparison[comparison["Year"] < 2019], comparison[comparison["Year"] == 2019]
        mu, sd, w, b = ridge_fit(tr[cols].to_numpy(float), tr["log_CHL"].to_numpy())
        pred = ridge_predict(va[cols].to_numpy(float), mu, sd, w, b)
        y = va["log_CHL"].to_numpy()
        r2 = 1 - ((y - pred)**2).sum() / ((y - y.mean())**2).sum()
        print("R² na validação:", round(r2, 3))

def auc_fold(df, ano, cols, make_model, so_completas=False, imputar=False):
    chl_real = df["CHL"].notna() # mask para linhas com um valor real de CHL

    tr = df["Year"] < ano # mask com True para os anos de treino
    if so_completas:
        tr = tr & chl_real # estratégia A -> treino com apenas valores reais de CHL
    va = (df["Year"] == ano) & chl_real  # validação: o ano, só linhas com CHL real

    d = df.copy() 
    # Estratégia B ->                       
    if imputar:                           
        for c in ["log_CHL", "CHL_uncertainty"]:
            d[c] = d[c].fillna(d.loc[tr, c].median())   # mediana só do treino

    # Matrizes de treino e validação, com as colunas necessárias
    X_tr = d.loc[tr, cols].to_numpy(float)   
    X_va = d.loc[va, cols].to_numpy(float)

    # Vetores com os valores (HAB) do treino e validação
    y_tr = d.loc[tr, "HAB"].to_numpy()
    y_va = d.loc[va, "HAB"].to_numpy()
    assert len(np.unique(y_va)) == 2, f"Ano {ano}: validação só tem uma classe"

    # Normalizar
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)   
    sd[sd == 0] = 1.0                              
    X_tr, X_va = (X_tr - mu) / sd, (X_va - mu) / sd

    modelo = make_model() # modelo é o modelo passado através do enunciado
    modelo.fit(X_tr, y_tr) # modelo é treinado                  
    probs = modelo.predict_proba(X_va)[:, 1] # probabilidade de HAB na validação
    return roc_auc_score(y_va, probs) # valor de AUC da validação

def main():
    df = pd.read_excel("data/classification_task1_train.xlsx")
    df = df.sort_values("Date").reset_index(drop=True)
    date = df["Date"].dt.dayofyear
    df["log_CHL"] = np.log(df["CHL"]) 

    # Append do ciclo do ano para garantir continuidade entre anos
    df["doy_sin"] = np.sin(2*np.pi*date/365)
    df["doy_cos"] = np.cos(2*np.pi*date/365)

    for col in [0, 1, 2, 3, 4, 5]:
        df[f"cluster_{col}"] = (df["spatial_cluster"] == col).astype(float)

    # Teste para verificar se o Ridge é bom para prever CHL (Resposta: não)
    teste_ridge_fit(df, run=False)

    # Definir variantes para teste de AUC
    base = ["doy_sin", "doy_cos", "SST", "UI"] + [f"cluster_{c}" for c in range(6)]
    com_chl = base + ["log_CHL", "CHL_uncertainty"] 
    variantes = {
        "A: so completas":      dict(cols=com_chl, so_completas=True),
        "B: imputa mediana":    dict(cols=com_chl, imputar=True),
        "C: sem CHL":           dict(cols=base),
    }   
    anos_val = [2016, 2018, 2019, 2020]
    # Teste de AUC para cada modelo
    for nome_modelo, model in modelos.items():
        print(f"Modelo: {nome_modelo}")
        for nome, cfg in variantes.items():
            aucs = [auc_fold(df, ano, make_model=model, **cfg) for ano in anos_val]
            print(f"{nome:25s}", np.round(aucs, 3), "média", round(np.mean(aucs), 3), "dp", round(np.std(aucs), 3))

if __name__ == "__main__":
    main()


