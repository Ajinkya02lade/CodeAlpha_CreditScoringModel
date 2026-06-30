# import pandas as pd
# import joblib

# # ===============================
# # Load Saved Model and Scaler
# # ===============================

# model = joblib.load("models/credit_model.pkl")
# scaler = joblib.load("models/scaler.pkl")

# # ===============================
# # Customer Details
# # (Change these values to test)
# # ===============================

# customer = pd.DataFrame({
#     "person_age": [30],
#     "person_income": [60000],
#     "person_home_ownership": [3],      # Encoded value
#     "person_emp_length": [5],
#     "loan_intent": [2],                # Encoded value
#     "loan_grade": [1],                 # Encoded value
#     "loan_amnt": [10000],
#     "loan_int_rate": [12.5],
#     "loan_percent_income": [0.16],
#     "cb_person_default_on_file": [0],  # N = 0, Y = 1
#     "cb_person_cred_hist_length": [4]
# })

# # ===============================
# # Scale Features
# # ===============================

# customer_scaled = scaler.transform(customer)

# # ===============================
# # Prediction
# # ===============================

# prediction = model.predict(customer_scaled)
# probability = model.predict_proba(customer_scaled)

# print("=" * 50)

# if prediction[0] == 0:
#     print("Prediction : LOW CREDIT RISK")
#     print("Loan Status: APPROVED")
# else:
#     print("Prediction : HIGH CREDIT RISK")
#     print("Loan Status: REJECTED")

# print("\nProbability")
# print(f"Good Credit : {probability[0][0]*100:.2f}%")
# print(f"Bad Credit  : {probability[0][1]*100:.2f}%")

# print("=" * 50)


import joblib
import pandas as pd

model = joblib.load("models/credit_model.pkl")

data = pd.DataFrame([{
    "person_age": 30,
    "person_income": 60000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 10000,
    "loan_int_rate": 12.5,
    "loan_percent_income": 0.2,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 4
}])

prediction = model.predict(data)

print("Loan Approved" if prediction[0] == 0 else "Loan Rejected")