# Flight Price Prediction (Linear Regression with Gradient Descent)

## Overview
This project:
1. Generates synthetic flight data (`generate_data.py`)
2. Cleans the data, removes outliers, and trains a Linear Regression model with Gradient Descent (`train_model.py`)
3. Deploys the model as an interactive Gradio app (`app.py`)

## Files
- `generate_data.py` → Creates synthetic flight data (`flights_data.csv`)
- `train_model.py` → Cleans data, trains model, saves `linear_model.pkl` and `scaler.pkl`
- `app.py` → Gradio interface to make predictions
- `requirements.txt` → Python dependencies
- `README.md` → Documentation

## Usage
1. Run `python generate_data.py` → generates dataset
2. Run `python train_model.py` → trains & saves model
3. Deploy `app.py` on Hugging Face Spaces

