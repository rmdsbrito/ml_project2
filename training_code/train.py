import pickle
import sys
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

# resolved project root directory, parents[1] is the project root (one level above this script)
RAIZ = Path(__file__).resolve().parents[1]

# shared with prediction time, so train and predict build features identically
sys.path.insert(0, str(RAIZ / "best_model"))
from utils import build_X, COLUNAS

# defined input dataset
DADOS = RAIZ / "data" / "classification_task1_train.xlsx"
SAIDA = RAIZ / "best_model" / "model.pkl"

# computes mean and std for each column; fit on training splits only to prevent data leakage
def normalizar_fit(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    
    return mu, sd

# run expanding-window temporal cross-validation across target years
def validar(df, X, y, make_model, anos):
    aucs, y_todos, p_todos = [], [], []
    
    for ano in anos:
        # train on all prior years; validate on non-null CHL rows of the target year
        tr = (df["Year"] < ano).to_numpy()
        va = ((df["Year"] == ano) & df["CHL"].notna()).to_numpy()
        
        # fit scaler strictly on the training fold before transforming
        mu, sd = normalizar_fit(X[tr])
        m = make_model().fit((X[tr] - mu) / sd, y[tr])
        p = m.predict_proba((X[va] - mu) / sd)[:, 1]
        
        aucs.append(roc_auc_score(y[va], p))
        y_todos.append(y[va])
        p_todos.append(p)
        
    return aucs, np.concatenate(y_todos), np.concatenate(p_todos)

# computes confusion matrix counts and classification metrics at a probability threshold
def metricas(y, p, limiar=0.5):
    pred = (p >= limiar).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=recall, precision=precision, f1=f1)


def main():
    # load dataset chronologically and extract feature matrix X and binary target y
    df = pd.read_excel(DADOS).sort_values("Date").reset_index(drop=True)
    X, y = build_X(df), df["HAB"].to_numpy()

    # candidate model factories
    modelos = {
        "logreg C=0.1": lambda: LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000), 
        "mlp (12 lay) a=0.1": lambda: MLPClassifier(hidden_layer_sizes=(12,), alpha=0.1, max_iter=2000, random_state=0),
    }

    # evaluate each candidate via temporal validation and log performance metrics
    for nome, make in modelos.items():
        aucs, y_v, p_v = validar(df, X, y, make, anos=[2016, 2018, 2019, 2020])
        m = metricas(y_v, p_v)
        
        print(f"{nome:20s} | AUC per year {np.round(aucs, 3)} mean {np.mean(aucs):.3f}")
        print(f"{'':20s} | recall {m['recall']:.2f} precision {m['precision']:.2f} f1 {m['f1']:.2f} "f"| TP {m['tp']} FP {m['fp']} FN {m['fn']} TN {m['tn']}")

    # final fit on full dataset (2013-2020): logistic regression is chosen over MLP for higher predictability and stability despite slightly lower validation scores
    make = modelos["logreg C=0.1"]
    mu, sd = normalizar_fit(X)
    modelo = make().fit((X - mu) / sd, y)

    # serialize model alongside scaler statistics and feature names for inference
    with open(SAIDA, "wb") as f:
        pickle.dump({"model": modelo, "mu": mu, "sd": sd, "colunas": COLUNAS}, f)
        
    print("model.pkl saved in", SAIDA)

if __name__ == "__main__":
    main()