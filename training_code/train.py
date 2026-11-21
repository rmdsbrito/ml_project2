import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

# Caminhos relativos a este ficheiro
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "best_model"))
from utils import build_X, COLUNAS

DADOS = RAIZ / "data" / "classification_task1_train.xlsx"
SAIDA = RAIZ / "best_model" / "model.pkl"

# Função para obter a média e sd dos vários arrays
def normalizar_fit(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd

# Função para validar o modelo num dataset do ano "ano" que não contém nenhum valor NaN
def validar(df, X, y, make_model, anos):
    aucs, y_todos, p_todos = [], [], []
    for ano in anos:
        tr = (df["Year"] < ano).to_numpy()
        va = ((df["Year"] == ano) & df["CHL"].notna()).to_numpy()
        mu, sd = normalizar_fit(X[tr])                      # só com o treino
        m = make_model().fit((X[tr] - mu) / sd, y[tr])
        p = m.predict_proba((X[va] - mu) / sd)[:, 1]
        aucs.append(roc_auc_score(y[va], p))
        y_todos.append(y[va])
        p_todos.append(p)
    return aucs, np.concatenate(y_todos), np.concatenate(p_todos)

# Função para calcular as métricas para comparação dos modelos
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
    df = pd.read_excel(DADOS).sort_values("Date").reset_index(drop=True)
    X, y = build_X(df), df["HAB"].to_numpy()

    # Modelos finais escolhidos
    modelos = {
        "logreg C=0.1": lambda: LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000),
        "mlp (12,) a=0.1": lambda: MLPClassifier(hidden_layer_sizes=(12,), alpha=0.1,
                                                 max_iter=2000, random_state=0),
    }

    # Ciclo para validação de ambos os modelos com datasets anuais de validação sem NaNs
    for nome, make in modelos.items():
        aucs, y_v, p_v = validar(df, X, y, make, anos=[2016, 2018, 2019, 2020])
        m = metricas(y_v, p_v)
        print(f"{nome:16s} AUC por ano {np.round(aucs, 3)} média {np.mean(aucs):.3f}")
        print(f"{'':16s} recall {m['recall']:.2f} precision {m['precision']:.2f} f1 {m['f1']:.2f} "
              f"| TP {m['tp']} FP {m['fp']} FN {m['fn']} TN {m['tn']}")

    # Fit final: logreg com todos os dados (2013-2020)
    # Ainda que o MLP dê resultados melhores, decidiu-se seguir com a logreg por ser mais previsível
    make = modelos["logreg C=0.1"]
    mu, sd = normalizar_fit(X)
    modelo = make().fit((X - mu) / sd, y)

    with open(SAIDA, "wb") as f:
        pickle.dump({"model": modelo, "mu": mu, "sd": sd, "colunas": COLUNAS}, f)
    print("model.pkl gravado em", SAIDA)

if __name__ == "__main__":
    main()