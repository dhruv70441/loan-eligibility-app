import os
import streamlit as st
import pandas as pd
from loan_approval_prediction.pipeline.prediction_pipeline import PredictionPipeline

st.set_page_config(page_title="Loan Eligibility Checker", layout="wide")
st.title("🏦 Loan Eligibility Prediction App")

# Load trained model
pp = PredictionPipeline(model_path="models/credit_risk_model.pkl")



# Create form for user input
with st.form(key='single_entry_form'):
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term (in months)", min_value=0)
    credit_history = st.selectbox("Credit History", [0, 1])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    customer = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }

    customer_df = pd.DataFrame([customer])  # convert to DataFrame
    preds, probs = pp.make_prediction(customer_df)

    st.success(f"Prediction: {'Approved' if preds[0]==1 else 'Rejected'}")
    st.info(f"Probability of Approval: {probs[0]*100:.2f}%")
