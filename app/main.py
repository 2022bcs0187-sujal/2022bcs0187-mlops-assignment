from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Constants (IMPORTANT for assignment)
NAME = "Sujal Chodvadiya"
ROLL_NO = "2022BCS0187"

app = FastAPI()

# Load model (make sure you save model in training)
MODEL_PATH = "model.joblib"

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

# Input schema (adjust based on your dataset)
class LoanInput(BaseModel):
    no_of_dependents: int
    education: int
    self_employed: int
    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float


# ✅ Health Check Endpoint
@app.get("/")
def health():
    return {
        "message": "API is running",
        "name": NAME,
        "roll_no": ROLL_NO
    }


# ✅ Prediction Endpoint
@app.post("/predict")
def predict(data: LoanInput):

    if model is None:
        return {"error": "Model not loaded"}

    input_data = pd.DataFrame([data.dict()])
    input_data.columns = input_data.columns.str.strip()

    input_data = pd.get_dummies(input_data)

    # Align columns
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_data)[0]

    return {
        "prediction": str(prediction),
        "name": "Sujal Chodvadiya",
        "roll_no": "2022BCS0187"
    }