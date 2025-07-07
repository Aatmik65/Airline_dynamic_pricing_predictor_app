import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Generate synthetic booking data for a route
def generate_route_history(origin, destination):
    start_date = datetime.today() - timedelta(days=180)
    dates = [start_date + timedelta(days=i) for i in range(180)]

    # Simulate demand (with weekly seasonality)
    demand = [
        100 + 20 * (date.weekday() in [4, 5]) + random.randint(-10, 10)
        for date in dates
    ]

    df = pd.DataFrame({
        'ds': dates,
        'y': demand
    })
    return df

# Forecast demand using Prophet
def forecast_demand(origin, destination):
    df = generate_route_history(origin, destination)
    model = Prophet(daily_seasonality=True, yearly_seasonality=False)
    model.fit(df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Create Plotly chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name='Historical Demand'))

    fig.update_layout(
        title=f"Demand Forecast: {origin} → {destination}",
        xaxis_title="Date",
        yaxis_title="Expected Bookings",
        height=400
    )

    return forecast, fig
