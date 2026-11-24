
import streamlit as st
import pandas as pd
import joblib
import json

# Page settings
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Load trained model and files
model = joblib.load("models/car_price_model.pkl")

with open("models/input_schema.json", "r") as file:
    schema = json.load(file)

with open("models/metrics.json", "r") as file:
    metrics = json.load(file)

# Website title
st.title("🚗 Car Price Prediction")
st.write("Enter your car details to predict its selling price.")

st.divider()

# Input fields
st.subheader("Enter Car Details")

car_name = st.text_input("Car Name", "Maruti Swift")
company = st.text_input("Company", "Maruti")
year = st.number_input("Year", min_value=1990, max_value=2026, value=2018)
kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=30000,
    step=1000
)
fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
)

# Predict button
if st.button("Predict Car Price"):
    input_data = {
        "name": car_name,
        "company": company,
        "year": year,
        "kms_driven": kms_driven,
        "fuel_type": fuel_type
    }

    input_df = pd.DataFrame(
        [[input_data[column] for column in schema["columns"]]],
        columns=schema["columns"]
    )

    prediction = model.predict(input_df)[0]

    st.success("Prediction completed!")
    st.subheader("Estimated Car Price")
    st.title("₹ " + format(prediction, ",.2f"))

# Model performance
st.divider()
st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{metrics['MAE']:,.0f}")
col2.metric("RMSE", f"{metrics['RMSE']:,.0f}")
col3.metric("R² Score", f"{metrics['R2']:.2f}")

st.caption("This is an estimated price. Actual market prices may vary.")
