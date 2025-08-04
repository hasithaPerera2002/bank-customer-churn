from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import numpy as np
import pandas as pd
import joblib

# Load model and preprocessors
scaler = joblib.load("scaler/scaler.pkl")
imputer = joblib.load("imputer/imputer.pkl")
feature_columns = joblib.load("feature_columns/feature_columns.pkl")

app = FastAPI(title="Churn Predictor")

class UserInput(BaseModel):
    CreditScore: float
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float
    Geography_Germany: float = 0
    Geography_Spain: float = 0
    Gender_Male: float = 0

@app.post("/predict")
def predict(input: UserInput,model_name: Literal[
    "decision_tree", "random_forest", "svc", "mlp", "ann", "logistic", "neural_network"
] ):
    

    try:
        model = joblib.load(f"models/model_{model_name}.pkl")
        imputer = joblib.load("imputer.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
    except FileNotFoundError as e:
        return {"error": str(e)}

    x = [input.data.get(col, 0) for col in feature_columns]
    x = np.array([x])

    x_imputed = imputer.transform(x)
    x_scaled = scaler.transform(x_imputed)

    prediction = model.predict(x_scaled)[0]
    return {"Exited": bool(prediction)}

@app.get("/metrics")
def get_metrics():
    models = ["decision_tree", "rf", "svc", "mlp", "ann", "logistic", "nn"]

    all_metrics = {}

    for model_name in models:
        try:
            df = pd.read_csv(f"scores/model_scores_{model_name}.csv")
            metrics = dict(zip(df["Metric"], df["Score"]))
            all_metrics[model_name] = metrics
        except Exception as e:
            all_metrics[model_name] = {"error": str(e)}

    return {"model_metrics": all_metrics}