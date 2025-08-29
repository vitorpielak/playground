# %%

import pandas as pd


df = pd.read_csv("Hearing.csv",delimiter=",",encoding="latin-1")
df.head()
df.info()
df.describe()
df.columns