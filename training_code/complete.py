import pandas as pd
import numpy as np
from sklearn import base
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# hyperparameter grid for candidate classifiers
modelos_teste = {
    "logreg C": (lambda u,v: LogisticRegression(C=u, class_weight="balanced", max_iter=1000), [0.01, 0.1, 1, 10, 100], None),
    
    "arvore depth": (lambda u,v: DecisionTreeClassifier(max_depth=u, min_samples_leaf=10,class_weight="balanced", random_state=0), [3, 4, 5, 6], None),
    
    "rf depth x leaf": (lambda u,v: RandomForestClassifier(n_estimators=300, max_depth=u, min_samples_leaf=v, class_weight="balanced", random_state=0), [3, 5, 7], [5, 10]),
    
    "mlp layers x alpha": (lambda u,v: MLPClassifier(hidden_layer_sizes=u, alpha=v, max_iter=2000, random_state=0), [(8,), (12,), (16,)], [1e-1, 1, 5, 10]),
    
    "kNN k x weights": (lambda u, v: KNeighborsClassifier(n_neighbors=u, weights=v),[5, 15, 30, 60], ["uniform", "distance"],),
    
    "naive bayes var_smoothing": (lambda u, v: GaussianNB(var_smoothing=u), [1e-9, 1e-3, 1e-1, 1], None),
}

# top-performing models shortlisted for final holdout comparison (2020)
modelos_finais = {
    "logreg (C=0.1)": lambda: LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000),
    
    "mlp (12 lay) a=0.1": lambda: MLPClassifier(hidden_layer_sizes=(12,), alpha=0.1, max_iter=2000, random_state=0),
    
    "kNN 60 nei - uniform": lambda: KNeighborsClassifier(n_neighbors=60, weights="uniform"),
}

# standardize features and fit closed-form L2-regularized linear regression
def ridge_fit(X, y, lam=1.0):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    
    # solve normal equations A = X^T X + lambda * I and A*w = X^T * (y - y_mean) regarding w
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    w = np.linalg.solve(A, Xs.T @ (y - y.mean()))
    return mu, sd, w, y.mean()

# predict target values using fitted standardization parameters and ridge weights
def ridge_predict(X, mu, sd, w, b):
    return ((X - mu) / sd) @ w + b

# evaluate whether log(CHL) can be linearly predicted from environmental/spatial features
def teste_ridge_fit(df, run=True):
    if run:
        comparison = df[df["CHL"].notna()].copy()
        comparison["log_CHL"] = np.log(comparison["CHL"]) 
    
        cols = ["SST", "UI", "doy_sin", "doy_cos"] + [f"cluster_{i}" for i in range(6)] 
        print(comparison[cols + ["log_CHL"]].corr()["log_CHL"].round(2))
        
        # temporal splits on dataset, train before 2019 and validate on 2019
        tr, va = comparison[comparison["Year"] < 2019], comparison[comparison["Year"] == 2019]
        mu, sd, w, b = ridge_fit(tr[cols].to_numpy(float), tr["log_CHL"].to_numpy())
        pred = ridge_predict(va[cols].to_numpy(float), mu, sd, w, b)
        
        y = va["log_CHL"].to_numpy()
        r2 = 1 - ((y - pred)**2).sum() / ((y - y.mean())**2).sum()
        print("R² in validation:", round(r2, 3))

# compute ROC-AUC on a single validation year using an expanding-window training split
def auc_calc(df, ano, cols, make_model, so_completas=False, imputar=False):
    chl_real = df["CHL"].notna()

    # train on all years prior to target year; optionally drop rows with missing CHL
    tr = df["Year"] < ano
    if so_completas:
        tr = tr & chl_real
        
    # validate strictly on observed-CHL rows of the target year for fair cross-strategy comparison
    va = (df["Year"] == ano) & chl_real

    d = df.copy() 
    # impute missing CHL features using training-fold medians only to prevent leakage                 
    if imputar:         
        for c in ["log_CHL", "CHL_uncertainty"]:
            d[c] = d[c].fillna(d.loc[tr, c].median())

    X_tr = d.loc[tr, cols].to_numpy(float)   
    X_va = d.loc[va, cols].to_numpy(float)
    y_tr = d.loc[tr, "HAB"].to_numpy()
    y_va = d.loc[va, "HAB"].to_numpy()
    assert len(np.unique(y_va)) == 2, f"Year {ano}: validation only has one class"

    # standardize features strictly using training statistics
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)   
    sd[sd == 0] = 1.0                              
    X_tr, X_va = (X_tr - mu) / sd, (X_va - mu) / sd

    modelo = make_model()
    modelo.fit(X_tr, y_tr)     
    
    # probability of having HAB on validation
    probs = modelo.predict_proba(X_va)[:, 1]
    
    return roc_auc_score(y_va, probs)

# print pearson correlations between candidate features and HAB prior to "ano" variable
def correlation(df, ano, PRINT=False):
    if PRINT:
        tr = df[df["Year"] < ano]
        cols_corr = ["doy_sin", "doy_cos", "SST", "UI"] + [f"cluster_{c}" for c in range(6)]
        print(tr[cols_corr + ["HAB"]].corr()["HAB"].drop("HAB").round(2))

# run feature removal study using logistic regression across validation years
def exp_features(df, anos=(2016, 2018, 2019), PRINT=False):
    if PRINT:
        base = ["doy_sin", "doy_cos", "SST", "UI"] + [f"cluster_{c}" for c in range(6)]
        
        variantes = {
            "base":                    base,
            "sem UI":                  [c for c in base if c != "UI"],
            "sem doy_cos":             [c for c in base if c != "doy_cos"],
            "sem UI nem doy_cos":      [c for c in base if c not in ("UI", "doy_cos")],
        }
        
        logreg = lambda: LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)
        
        for nome, cols in variantes.items():
            aucs = [auc_calc(df, ano, make_model=logreg, cols=cols) for ano in anos]
            
            print(f"{nome:22s} {np.round(aucs, 3)} mean {np.mean(aucs):.3f} dp {np.std(aucs):.3f}")

def main():
    # load and sort dataset chronologically
    df = pd.read_excel("data/classification_task1_train.xlsx")
    df = df.sort_values("Date").reset_index(drop=True)
    
    # log-transform CHL, cyclical day-of-year encoding, and one-hot spatial clusters
    date = df["Date"].dt.dayofyear
    df["log_CHL"] = np.log(df["CHL"])
    df["doy_sin"] = np.sin(2*np.pi*date/365)
    df["doy_cos"] = np.cos(2*np.pi*date/365)

    for col in [0, 1, 2, 3, 4, 5]:
        df[f"cluster_{col}"] = (df["spatial_cluster"] == col).astype(float)

    # verify whether ridge regression can reliably impute missing CHL (no)
    teste_ridge_fit(df, run=False)

    # define missing-CHL handling strategies, we decided not to use CHL (option C)
    base = ["doy_sin", "doy_cos", "SST", "UI"] + [f"cluster_{c}" for c in range(6)]
    com_chl = base + ["log_CHL", "CHL_uncertainty"] 
    variantes = {
        "A: so completas":      dict(cols=com_chl, so_completas=True),
        "B: imputa mediana":    dict(cols=com_chl, imputar=True),
        "C: sem CHL":           dict(cols=base),
    }   
    variante_escolhida = variantes["C: sem CHL"]
    anos_val = [2016, 2018, 2019]
    
    # grid search across candidate algorithms and hyperparameters
    PRINT_MODELS = False
    if PRINT_MODELS:
        for nome_modelo, (model, valor1, valor2) in modelos_teste.items():
            
            print(f"Model: {nome_modelo}")
            
            for v1 in valor1:
                for v2 in (valor2 if valor2 is not None else [None]):
                    make = lambda u=v1, v=v2: model(u, v)
                    aucs = [auc_calc(df, ano, make_model=make, **variante_escolhida) for ano in anos_val]
                    
                    print(f"  {str(v1):<8} {str(v2):<8}", "Annual AUC:", np.round(aucs, 3), "AUC Mean:", round(np.mean(aucs), 3), "AUC Std Dev:", round(np.std(aucs), 3))

    # evaluate shortlisted models on the 2020 holdout year
    print("\n\nFinal validation values (2020) for the best models: \n")
    for nome, make in modelos_finais.items():
        auc = auc_calc(df, 2020, make_model=make, **variante_escolhida)
        
        print(f"{nome:25s} 2020 AUC = {auc:.3f}")

    # optional exploratory correlation and feature removal diagnostics
    correlation(df, 2020, PRINT=False)
    exp_features(df=df, PRINT=False)

if __name__ == "__main__":
    main()