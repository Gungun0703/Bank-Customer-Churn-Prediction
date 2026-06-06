import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Bank Customer Churn Prediction")

credit_score = st.number_input("Credit Score", 300, 900, 600)
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", 18, 100, 40)
tenure = st.number_input("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 300000.0, 60000.0)
products = st.number_input("Number of Products", 1, 4, 2)
has_card = st.selectbox("Has Credit Card", [0, 1])
active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, 300000.0, 50000.0)

country = st.selectbox("Country", ["France", "Germany", "Spain"])

geo_germany = 1 if country == "Germany" else 0
geo_spain = 1 if country == "Spain" else 0

gender = 1 if gender == "Male" else 0

if st.button("Predict"):

    data = [[
        credit_score,
        gender,
        age,
        tenure,
        balance,
        products,
        has_card,
        active,
        salary,
        geo_germany,
        geo_spain
    ]]

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")