import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Smart City Traffic Prediction",
    page_icon="🚦"
)

# Title
st.title("🚦 Smart City Traffic Prediction")
st.write("Predict the expected number of vehicles based on Junction, Date and Time.")

# Load trained model
model = joblib.load("traffic_model.pkl")

# User inputs
st.subheader("Enter Traffic Details")

junction = st.selectbox(
    "Select Junction",
    [1, 2, 3, 4]
)

date_input = st.date_input("Select Date")

time_input = st.time_input("Select Time")

# Prediction button
if st.button("Predict Traffic 🚗"):

    # Combine date and time
    input_datetime = pd.to_datetime(
        str(date_input) + " " + str(time_input)
    )

    # Create features
    hour = input_datetime.hour
    day = input_datetime.day
    month = input_datetime.month
    year = input_datetime.year
    day_of_week = input_datetime.dayofweek
    is_weekend = int(day_of_week >= 5)

    # Create DataFrame
    input_data = pd.DataFrame([{
        "Junction": junction,
        "Hour": hour,
        "Day": day,
        "Month": month,
        "Year": year,
        "DayOfWeek": day_of_week,
        "IsWeekend": is_weekend
    }])

    # Prediction
    prediction = model.predict(input_data)[0]
    prediction = round(prediction)

    # Show result
    st.success(f"🚗 Predicted Number of Vehicles: {prediction}")

    # Traffic level
    if prediction < 30:
        st.success("🟢 Traffic Level: LOW")
    elif prediction < 70:
        st.warning("🟡 Traffic Level: MEDIUM")
    else:
        st.error("🔴 Traffic Level: HIGH")