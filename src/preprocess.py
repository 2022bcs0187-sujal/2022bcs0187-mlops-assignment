import pandas as pd
import os

# Load data (use versioned file)
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "../data/loan_approval_dataset.csv")
df = pd.read_csv(file_path)

# Basic cleaning
df = df.dropna()

df.columns = df.columns.str.strip()

# Separate target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Encode categorical
X = pd.get_dummies(X)

# Save processed data
os.makedirs("data/processed", exist_ok=True)

X.to_csv("data/processed/X.csv", index=False)
y.to_csv("data/processed/y.csv", index=False)

print("Preprocessing done!")