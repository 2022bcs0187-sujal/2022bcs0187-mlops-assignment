import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import joblib


# Load processed data
X = pd.read_csv("data/processed/X.csv")
y = pd.read_csv("data/processed/y.csv").squeeze()

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MLflow
mlflow.set_experiment("2022bcs0187_experiment")

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model.joblib")

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    # Log MLflow
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

    # 🔥 REQUIRED: Save metrics.json (WITH NAME + ROLL NO)
    os.makedirs("reports", exist_ok=True)

    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "name": "Sujal Chodvadiya",
        "roll_no": "2022bcs0187"
    }

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f)

    print("Training complete!")