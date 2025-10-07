import os
import sys

# Get the absolute path of the project root (loan_app)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the project root to sys.path (Python’s import search list)
if project_root not in sys.path:
    sys.path.append(project_root)


import streamlit as st
import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline

st.set_page_config(page_title="Loan Eligibility Checker", layout="wide")
st.title("🏦 Loan Eligibility Prediction App")

# Load trained model
pp = PredictionPipeline(model_path="models/credit_risk_model.pkl")

st.sidebar.header("Prediction Mode")
mode = st.sidebar.selectbox("Select Mode", ["Single Entry", "Bulk CSV Upload"])

if mode == "Single Entry":
    st.subheader("Predict for a Single Customer")

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

elif mode == "Bulk CSV Upload":
    st.subheader("Upload CSV for Multiple Customers")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df_bulk = pd.read_csv(uploaded_file)
        st.dataframe(df_bulk.head())

        preds, probs = pp.make_prediction(df_bulk)
        preds_df = df_bulk.copy()
        preds_df['Prediction'] = preds
        preds_df['Approval_Probability'] = probs

        st.subheader("Prediction Results")
        st.dataframe(preds_df)

        csv = preds_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv'
        )
