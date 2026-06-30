import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Model
# ==========================

model = joblib.load("models/credit_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Scoring Model")
st.write("Predict whether a loan applicant is Low Risk or High Risk.")

st.markdown("---")

# ==========================
# User Inputs
# ==========================

age = st.number_input("Age", 18, 100, 30)

income = st.number_input("Annual Income", 1000, 1000000, 60000)

home = st.selectbox(
    "Home Ownership",
    ["MORTGAGE", "OWN", "RENT", "OTHER"]
)

emp = st.number_input(
    "Employment Length (Years)",
    0.0,
    50.0,
    5.0
)

loan_intent = st.selectbox(
    "Loan Purpose",
    [
        "DEBTCONSOLIDATION",
        "EDUCATION",
        "HOMEIMPROVEMENT",
        "MEDICAL",
        "PERSONAL",
        "VENTURE"
    ]
)

loan_grade = st.selectbox(
    "Loan Grade",
    ["A","B","C","D","E","F","G"]
)

loan_amount = st.number_input(
    "Loan Amount",
    500,
    100000,
    10000
)

interest = st.number_input(
    "Interest Rate",
    1.0,
    40.0,
    12.5
)

loan_percent_income = st.slider(
    "Loan Percent Income",
    0.0,
    1.0,
    0.20
)

default = st.selectbox(
    "Previous Default",
    ["N","Y"]
)

credit_history = st.number_input(
    "Credit History Length",
    1,
    30,
    4
)

# ==========================
# Encoding
# (Matches your LabelEncoder output)
# ==========================

home_map = {
    "MORTGAGE":0,
    "OTHER":1,
    "OWN":2,
    "RENT":3
}

intent_map = {
    "DEBTCONSOLIDATION":0,
    "EDUCATION":1,
    "HOMEIMPROVEMENT":2,
    "MEDICAL":3,
    "PERSONAL":4,
    "VENTURE":5
}

grade_map = {
    "A":0,
    "B":1,
    "C":2,
    "D":3,
    "E":4,
    "F":5,
    "G":6
}

default_map = {
    "N":0,
    "Y":1
}

# ==========================
# Prediction
# ==========================

if st.button("Predict"):

    customer = pd.DataFrame({
        "person_age":[age],
        "person_income":[income],
        "person_home_ownership":[home_map[home]],
        "person_emp_length":[emp],
        "loan_intent":[intent_map[loan_intent]],
        "loan_grade":[grade_map[loan_grade]],
        "loan_amnt":[loan_amount],
        "loan_int_rate":[interest],
        "loan_percent_income":[loan_percent_income],
        "cb_person_default_on_file":[default_map[default]],
        "cb_person_cred_hist_length":[credit_history]
    })

    customer_scaled = scaler.transform(customer)

    prediction = model.predict(customer_scaled)

    probability = model.predict_proba(customer_scaled)

    st.markdown("---")

    if prediction[0] == 0:
        st.success("✅ LOW CREDIT RISK")
        st.success("Loan Approved")
    else:
        st.error("❌ HIGH CREDIT RISK")
        st.error("Loan Rejected")

    st.write(f"### Good Credit Probability : {probability[0][0]*100:.2f}%")
    st.write(f"### Bad Credit Probability : {probability[0][1]*100:.2f}%")