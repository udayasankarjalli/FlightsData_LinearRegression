import gradio as gr
import numpy as np
import joblib
import os

# Load trained model and scaler
if not os.path.exists("scaler.pkl") or not os.path.exists("theta.npy"):
    raise FileNotFoundError(
        "Model files not found! Please run train_model.py first to generate scaler.pkl and theta.npy."
    )

scaler = joblib.load("scaler.pkl")
theta = np.load("theta.npy")

# Prediction function
def predict_price(flight_duration, distance, seats_booked, fuel_cost, airline_rating):
    # Prepare features as numpy array
    features = np.array([[flight_duration, distance, seats_booked, fuel_cost, airline_rating]])
    
    # Scale using the same scaler from training
    features_scaled = scaler.transform(features)

    # Add bias column (1), same as during training
    features_scaled = np.hstack((np.ones((features_scaled.shape[0], 1)), features_scaled))

    # Predict price
    predicted_price = np.dot(features_scaled, theta)

    return f"Predicted Flight Price: ₹{predicted_price[0][0]:.2f}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# ✈️ Flight Price Predictor")
    gr.Markdown("Enter flight details to estimate the price:")

    with gr.Row():
        flight_duration = gr.Number(label="Flight Duration (hours)", value=5.0)
        distance = gr.Number(label="Distance (km)", value=2500.0)

    with gr.Row():
        seats_booked = gr.Number(label="Seats Booked", value=150)
        fuel_cost = gr.Number(label="Fuel Cost", value=5000.0)

    airline_rating = gr.Slider(1, 10, value=7, step=1, label="Airline Rating (1-10)")

    predict_btn = gr.Button("Predict Price")
    output = gr.Textbox(label="Prediction")

    predict_btn.click(
        predict_price,
        inputs=[flight_duration, distance, seats_booked, fuel_cost, airline_rating],
        outputs=output
    )

# Launch app
if __name__ == "__main__":
    demo.launch()
