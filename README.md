---
title: Flight Price Predictor
emoji: ✈️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.47.0"
app_file: app.py
pinned: false
---


# ✈️ Flight Price Predictor

This project demonstrates how to train and deploy a **Linear Regression model** for predicting flight prices using **distance, duration, and fuel cost** as input features.  
The model is trained with synthetic/generated flight data and deployed on **Hugging Face Spaces** using a **Gradio interface**.

---

## 🚀 Features
- **Model Training**:  
  - Generates synthetic flight data (`generate_data.py`).  
  - Trains a custom Linear Regression model (`train_model.py`).  
  - Saves trained weights (`theta.npy`) and scaler (`scaler.pkl`).  

- **Deployment**:  
  - Interactive Gradio app (`app.py`).  
  - GitHub Actions CI/CD pipeline auto-deploys updates to Hugging Face Space.  

- **Prediction**:  
  - Input flight details in the UI.  
  - Get real-time predicted price instantly.

---

## 📂 Repository Structure
