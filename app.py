
import streamlit as st
import pandas as pd
from pricing_engine import predict_price
from forecasting import forecast_demand
from api_utils import get_weather_score, is_public_holiday, get_competitor_price

st.set_page_config(page_title="Airline Dynamic Pricing", layout="wide")
st.title("✈️ Airline Dynamic Pricing (India)")

# Input Section
st.sidebar.header("Flight Details")
origin = st.sidebar.selectbox("Origin City", ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Kolkata", "Chennai"])
destination = st.sidebar.selectbox("Destination City", ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Kolkata", "Chennai"])
while destination == origin:
    destination = st.sidebar.selectbox("Destination City", ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Kolkata", "Chennai"], index=1)
days_to_departure = st.sidebar.slider("Days Until Departure", 1, 60, 15)
time_of_day = st.sidebar.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
user_type = st.sidebar.selectbox("User Type", ["Business", "Leisure"])

# API-based features
weather_score = get_weather_score(origin)
holiday_flag = is_public_holiday(days_to_departure)
competitor_price = get_competitor_price(origin, destination)

# Prediction
our_price = predict_price(origin, destination, days_to_departure, time_of_day, user_type, weather_score, holiday_flag, competitor_price)

# UI Output
st.subheader("📈 Predicted Price")
st.metric("Our Price", f"₹{our_price:.2f}")
st.metric("Competitor Price", f"₹{competitor_price:.2f}")
st.metric("Weather Score", f"{weather_score:.2f}")
st.metric("Holiday Effect", "Yes" if holiday_flag else "No")

# Demand Forecast Chart
st.subheader("📊 30-Day Demand Forecast")
forecast_df, fig = forecast_demand(origin, destination)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit · Powered by XGBoost + Prophet")
