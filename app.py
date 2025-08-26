from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import numpy as np
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Load model and preprocessors
scaler = joblib.load("scaler/scaler.pkl")
imputer = joblib.load("imputer/imputer.pkl")
feature_columns = joblib.load("feature_columns/feature_columns.pkl")

app = FastAPI(title="Churn Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust to your React dev server URL
    allow_credentials=True,
    allow_methods=["*"],   # allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],   # allows Content-Type, Authorization, etc.
)

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
        imputer = joblib.load("imputer/imputer.pkl")
        scaler = joblib.load("scaler/scaler.pkl")
        feature_columns = joblib.load("feature_columns/feature_columns.pkl")
    except FileNotFoundError as e:
        print(f"Error loading model or preprocessors: {e}")
        return {"error": str(e)}
    
    print(f"Using model: {model_name}")
    print(f"User Input",input)

    x = [input.dict().get(col, 0) for col in feature_columns]
    x = np.array([x])

    x_imputed = imputer.transform(x)
    x_scaled = scaler.transform(x_imputed)

    prediction = model.predict(x_scaled)[0]
    print(f"Prediction: {prediction}")
    return {"Exited": bool(prediction)}

@app.get("/metrics")
def get_metrics():
    try:
        df = pd.read_csv("scores/model_scores_ann.csv")
        metrics = dict(zip(df["Metric"], df["Score"]))
        return {"Model Evaluation Scores": metrics}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)