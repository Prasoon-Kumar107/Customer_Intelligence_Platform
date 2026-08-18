import pickle as p
import pandas as pd
from src.data_preprocessing import data_preprocess
from src.feature_engineering import feature_engineering
from src.config import MODEL_FILE, BEST_THRESHOLD

with open(MODEL_FILE, "rb") as f:
    model = p.load(f)

def predict_churn(customer_data):
    data = pd.DataFrame([customer_data])
    data = data_preprocess(data)
    data = feature_engineering(data)
    churn_probability = model.predict_proba(data)[:, 1][0]
    prediction = int(churn_probability >= BEST_THRESHOLD)
    if prediction == 1:
        result = "Will Churn"
    else:
        result = "Won't Churn"
    return {
    "prediction": result,
    "churn_probability": float(round(churn_probability,2)),
    "threshold": BEST_THRESHOLD
    }