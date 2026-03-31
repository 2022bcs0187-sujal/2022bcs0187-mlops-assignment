import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn

# ==============================
# CONFIG (CHANGE PER RUN)
# ==============================
DATA_PATH = "data/loan_approval_dataset.csv"
MODEL_TYPE = "rf"
N_ESTIMATORS = 100
FEATURE_SET = "reduced"

NAME = "Sujal Chodvadiya"
ROLL_NO = "2022BCS0187"

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(DATA_PATH)
df = df.dropna()

# ==============================
# FEATURE SELECTION
# ==============================
all_features = [
    " no_of_dependents",
    " education",
    " self_employed",
    " income_annum",
    " loan_amount",
    " loan_term",
    " cibil_score",
    " residential_assets_value",
    " commercial_assets_value",
    " luxury_assets_value",
    " bank_asset_value"
]

reduced_features = [
    " income_annum",
    " loan_amount",
    " cibil_score",
    " bank_asset_value"
]

if FEATURE_SET == "all":
    selected_features = all_features
else:
    selected_features = reduced_features

X = df[selected_features]
y = df[" loan_status"]

# Encode categorical
X = pd.get_dummies(X)

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ==============================
# MODEL SELECTION
# ==============================
if MODEL_TYPE == "rf":
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS)
elif MODEL_TYPE == "lr":
    model = LogisticRegression(max_iter=1000)
else:
    raise ValueError("Invalid model")

# ==============================
# MLFLOW
# ==============================
mlflow.set_experiment("2022BCS0187_experiment")

with mlflow.start_run():

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    # Log params
    mlflow.log_param("dataset", DATA_PATH)
    mlflow.log_param("model", MODEL_TYPE)
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("feature_set", FEATURE_SET)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Save model for API
    joblib.dump(model, "model.joblib")

    # Save metrics.json (for CI)
    os.makedirs("reports", exist_ok=True)

    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "name": NAME,
        "roll_no": ROLL_NO
    }

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f)

    print("Run complete!")