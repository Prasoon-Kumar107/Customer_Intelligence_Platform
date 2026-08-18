import pandas as pd
import io
import matplotlib.pyplot as plt
from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse,StreamingResponse
from pydantic import BaseModel,Field
from typing import Annotated,Literal
from src.predict import predict_churn
from src.explain import waterfall_plot

app = FastAPI(title="Customer Churn Prediction API",description="API for predicting customer churn using XGBoost.",version="1.0.0")

class CustomerData(BaseModel):
    gender: Annotated[Literal["Male","Female"],Field(description="Enter Customer's gender.")]
    SeniorCitizen: Annotated[Literal[0,1],Field(description="Enter 1 if customer is Senior Citizen and 0 if not.")]
    Partner: Annotated[Literal["Yes","No"],Field(description="Does Customer have a Partner?")]
    Dependents: Annotated[Literal["Yes","No"],Field(description="Does Customer have any Dependents on them?")]
    tenure: Annotated[float,Field(description="Enter for how many months customer have been a member?",ge=0,le=100,)]
    PhoneService: Annotated[Literal["Yes","No"],Field(description="Does Customer have Phone Service?")]
    MultipleLines: Annotated[Literal["Yes","No","No phone service"],Field(description="Does Customer have MultipleLines?")]
    InternetService: Annotated[Literal['DSL', 'Fiber optic', 'No'],Field(description="Which type of InternetService Customer have?")]
    OnlineSecurity: Annotated[Literal['No', 'Yes', 'No internet service'],Field(description="Does Customer have Online Security?")]
    OnlineBackup: Annotated[Literal['Yes', 'No', 'No internet service'],Field(description="Does Customer have Online Backup?")]
    DeviceProtection: Annotated[Literal['No', 'Yes', 'No internet service'],Field(description="Does Customer have Device Protection?")]
    TechSupport: Annotated[Literal['No', 'Yes', 'No internet service'],Field(description="Does Customer have Tech Support?")]
    StreamingTV: Annotated[Literal['No', 'Yes', 'No internet service'],Field(description="Does Customer Stream TV?")]
    StreamingMovies: Annotated[Literal['No', 'Yes', 'No internet service'],Field(description="Does Customer Stream Movies?")]
    Contract: Annotated[Literal['Month-to-month', 'One year', 'Two year'],Field(description="What type of Contract Customer have?")]
    PaperlessBilling: Annotated[Literal['No', 'Yes'],Field(description="Does Customer do Paperless Billing?")]
    PaymentMethod: Annotated[Literal['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'],Field(description="What Payment Method does Customer Use?")]
    MonthlyCharges: Annotated[float,Field(description="What is the monthly charge payed by Customer?",gt=0,lt=200)]
    TotalCharges: Annotated[float,Field(description="What is the total charge payed by Customer?",gt=0,lt=10000)]

@app.get("/")
def home():
    return {
        "message" : "Welcome to api of Churn Predictor"
    }

@app.get("/health")
def health_check():
    return {
        "status" : "Ok"
    }

@app.post("/predict")
def predict(customer: CustomerData):
    try:
        result = predict_churn(customer.model_dump())
        return JSONResponse(status_code=200,content=result)
    except Exception:
        raise HTTPException(status_code=500,detail="Prediction failed.")

@app.post("/explain")
def explain(customer_data: CustomerData):
    try:
        input_df = pd.DataFrame([customer_data.model_dump()])
        fig = waterfall_plot(input_df)
        buffer = io.BytesIO()
        fig.savefig(buffer,format="png",bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        return StreamingResponse(buffer,media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))