import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("loan_model.pkl")

st.title("🏦 Automated Loan Eligibility Predictor")
st.write("Enter applicant details to check loan approval status.")

# Input fields
gender = st.selectbox("Gender", ["Female", "Male"])
married = st.selectbox("Married", ["No", "Yes"])
education = st.selectbox("Education", ["Not Graduate", "Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.selectbox("Loan Amount Term (months)", [120, 180, 240, 300, 360])
credit_history = st.selectbox("Credit History", ["No", "Yes"])
property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

if st.button("Predict Loan Eligibility"):
    # Convert categorical inputs to numeric (same encoding as training)
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    education_val = 1 if education == "Graduate" else 0
    self_employed_val = 1 if self_employed == "Yes" else 0
    credit_val = 1 if credit_history == "Yes" else 0
    property_val = {"Rural":0, "Semiurban":1, "Urban":2}[property_area]

    features = np.array([[
        gender_val, married_val, education_val, self_employed_val,
        applicant_income, coapplicant_income,
        loan_amount, loan_term, credit_val, property_val
    ]])

    prediction = model.predict(features)[0]
    result = "✅ Eligible for Loan" if prediction == 1 else "❌ Not Eligible for Loan"
    st.subheader(result)
