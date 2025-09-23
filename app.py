import gradio as gr
import joblib
import numpy as np

# Load model & scaler
model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")

# Prediction function
def predict_price(flight_duration, distance, seats_booked, fuel_cost, airline_rating):
    features = np.array([[flight_duration, distance, seats_booked, fuel_cost, airline_rating]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    return f"Predicted Flight Price: {prediction[0]:.2f}"

# Gradio Interface
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Flight Duration (hrs)"),
        gr.Number(label="Distance (km)"),
        gr.Number(label="Seats Booked"),
        gr.Number(label="Fuel Cost"),
        gr.Number(label="Airline Rating (1-10)")
    ],
    outputs="text",
    title="Flight Price Prediction (Linear Regression with Gradient Descent)"
)

if __name__ == "__main__":
    iface.launch()
