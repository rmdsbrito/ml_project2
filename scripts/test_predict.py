import sys
from pathlib import Path

import numpy as np
import pandas as pd

# resolve project root directory, parents[1] is the project root (one level above this script)
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "best_model"))    # só neste script de teste
import my_model

# load dataset without target column to simulate unseen inference inputs
df = pd.read_excel(RAIZ / "data" / "classification_task1_train.xlsx")
df = df.drop(columns=["HAB"])
antes = df.copy()

# load serialized model and verify both batch and single-row inference
m = my_model.load_model(RAIZ / "best_model" / "model.pkl")
p = my_model.predict(m, df)
q = my_model.predict(m, df.iloc[[0]])

# check output shapes, finite probability bounds, and input immutability
print("all:", p.shape, "| one:", q.shape)
print("min", p.min().round(3), "max", p.max().round(3), "| finite:", np.isfinite(p).all())

assert df.equals(antes), "predicted changed input's DataFrame!"

print("OK: untouched input")