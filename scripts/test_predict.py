import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "best_model"))    # só neste script de teste
import my_model

df = pd.read_excel(RAIZ / "data" / "classification_task1_train.xlsx")
df = df.drop(columns=["HAB"])                   # o teste real não traz o alvo
antes = df.copy()

m = my_model.load_model(RAIZ / "best_model" / "model.pkl")
p = my_model.predict(m, df)                     # todas as linhas
q = my_model.predict(m, df.iloc[[0]])           # uma só linha

print("todas:", p.shape, "| uma:", q.shape)
print("min", p.min().round(3), "max", p.max().round(3), "| finito:", np.isfinite(p).all())
assert df.equals(antes), "o predict alterou o DataFrame de entrada!"
print("OK: input intacto")