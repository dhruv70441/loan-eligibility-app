from fastapi import FastAPI
from loan_eligibility.api.schemas import LoanApplication
from loan_eligibility.pipeline.prediction_pipeline import PredictionPipeline
import pandas as pd

app = FastAPI(
    title="Loan Eligibility API",
    version="1.0.0"
)


# GET api check endpoint
@app.get("/health")
def health_chechk():
    return {"status":"ok"}


# GET load model on app startup 
@app.on_event("startup")
def load_model():
    """
    Load model once when API starts
    """
    app.state.pipeline = PredictionPipeline()
    print("Prediction pipeline loaded")


# POST endpoint to make prediction
@app.post("/predict")
def predict_loan(application: LoanApplication):
    """
    Make loan eligibility prediction
    """
    data = pd.DataFrame([application.dict()])

    pipeline = app.state.pipeline

    prediction, probability = pipeline.make_prediction(data)
    
    return {
        "loan_approved": bool(prediction[0]),
        "approval_probability": float(probability[0])
    }
