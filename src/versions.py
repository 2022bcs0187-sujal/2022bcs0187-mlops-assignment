import pandas as pd

df = pd.read_csv("data/loan_approval_dataset.csv")
df_small = df.sample(frac=0.5)

df_small.to_csv("data/loan_approval_dataset_v1.csv", index=False)