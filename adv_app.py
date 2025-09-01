import streamlit as st
import pickle
import numpy as np

# Load the saved model (make sure you trained & saved it as 'treatment_model.sav')
model = pickle.load(open('lr.sav', 'rb'))

st.title("Hospital Treatment Cost Prediction App")

st.write("Enter patient details to predict estimated treatment cost.")

# Example Input Features (replace with actual columns from your dataset)
age = st.number_input("Patient Age", min_value=0, max_value=120, value=40)
days = st.number_input("Number of Hospital Days", min_value=1, max_value=365, value=5)
complications = st.number_input("Complications (0=No, 1=Yes)", min_value=0, max_value=1, value=0)
tests = st.number_input("Number of Medical Tests", min_value=0, max_value=50, value=5)
medicines = st.number_input("Medicine Cost Estimate", min_value=0.0, value=1000.0)

# Make prediction
if st.button("Predict Treatment Cost"):
    input_data = np.array([[age, days, complications, tests, medicines]])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Treatment Cost: ₹{prediction:,.2f}")

