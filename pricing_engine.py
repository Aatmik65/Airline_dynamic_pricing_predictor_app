import pandas as pd
import pickle
import os

# Path to the trained XGBoost model
MODEL_PATH = os.path.join("model", "demand_model.pkl")

def load_model():
    # Load the pre-trained model from the model folder
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model

def predict_price(origin, destination, days_to_departure, time_of_day, user_type, weather_score, holiday_flag, competitor_price):
    # Load model
    model = load_model()

    # Encode inputs
    time_map = {"Morning": 1, "Afternoon": 2, "Evening": 3}
    user_map = {"Frequent Flyer": 1, "Casual Traveler": 0}

    # Create input dataframe
    X = pd.DataFrame([[
        days_to_departure,
        time_map[time_of_day],
        user_map[user_type],
        weather_score,
        holiday_flag,
        competitor_price
    ]], columns=[
        "days_to_departure",
        "time_of_day",
        "user_type",
        "weather_score",
        "holiday_flag",
        "competitor_price"
    ])

    # Predict price
    return model.predict(X)[0]
