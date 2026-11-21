import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import DADOS, build_X, validar

df = pd.read_excel(DADOS).sort_values("Date").reset_index(drop=True)
X, y = build_X(df), df["HAB"].to_numpy()
anos = [2016, 2018, 2019, 2020]

# Referência: a logreg (não depende de seed)
aucs_lr, _, _ = validar(df, X, y,
                        lambda: LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000),
                        anos)
print(f"logreg        {np.round(aucs_lr, 3)} média {np.mean(aucs_lr):.3f}")

# MLP com 5 seeds diferentes
medias = []
for seed in range(5):
    make = lambda s=seed: MLPClassifier(hidden_layer_sizes=(12,), alpha=0.1,
                                        max_iter=2000, random_state=s)
    aucs, _, _ = validar(df, X, y, make, anos)
    medias.append(np.mean(aucs))
    print(f"mlp seed={seed}   {np.round(aucs, 3)} média {np.mean(aucs):.3f}")

print(f"\nMLP: média das seeds {np.mean(medias):.3f}  min {np.min(medias):.3f}  max {np.max(medias):.3f}")
print(f"logreg: {np.mean(aucs_lr):.3f}")