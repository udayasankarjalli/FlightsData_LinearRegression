# ----------------------------
# Step 0: Import Libraries
# ----------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# 1. Load Data
data = pd.read_csv("flights_data.csv")
data = pd.DataFrame(data)

# Example: assuming df is your DataFrame
# Fill missing values
data['flight_duration'].fillna(data['flight_duration'].median(), inplace=True)
data['fuel_cost'].fillna(data['fuel_cost'].median(), inplace=True)
print(data.info())

# ----------------------------
# Outlier Removal using IQR
# ----------------------------

Q1 = data['price'].quantile(0.25)
Q3 = data['price'].quantile(0.75)
IQR = Q3 - Q1

# Keep only rows where price is within 1.5*IQR
data_clean = data[(data['price'] >= Q1 - 1.5*IQR) & (data['price'] <= Q3 + 1.5*IQR)]

print("Original data shape:", data.shape)
print("Data shape after removing outliers:", data_clean.shape)

# ----------------------------
# Step 1: Prepare Features and Target
# ----------------------------
# Assuming you already have a cleaned DataFrame 'data_clean'
X = data_clean[['flight_duration', 'distance', 'seats_booked', 'fuel_cost', 'airline_rating']].values
y = data_clean['price'].values.reshape(-1, 1)

# Feature scaling (important for gradient descent)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add bias term (column of ones)
X_scaled = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ----------------------------
# Step 2: Gradient Descent Implementation
# ----------------------------
def gradient_descent(X, y, lr=0.01, epochs=5000):
    m, n = X.shape
    theta = np.zeros((n, 1))  # Initialize weights
    cost_history = []

    for i in range(epochs):
        y_pred = np.dot(X, theta)
        error = y_pred - y
        cost = (1/(2*m)) * np.sum(error**2)
        cost_history.append(cost)

        # Gradient descent update
        theta -= (lr/m) * np.dot(X.T, error)

        # Optional: print cost every 500 epochs
        if i % 500 == 0:
            print(f"Epoch {i}, Cost: {cost:.2f}")

    return theta, cost_history

# ----------------------------
# Step 3: Train the Model
# ----------------------------
theta, cost_history = gradient_descent(X_train, y_train, lr=0.01, epochs=5000)

# ----------------------------
# Step 4: Predictions
# ----------------------------
y_pred_train = np.dot(X_train, theta)
y_pred_test = np.dot(X_test, theta)

# ----------------------------
# Step 5: Evaluate Model
# ----------------------------
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\nGradient Descent Linear Regression Evaluation:")
print(f"Train R²: {r2_train:.3f}")
print(f"Test R²: {r2_test:.3f}")
print(f"Test MAE: {mae_test:.2f}")
print(f"Test RMSE: {rmse_test:.2f}")

# ----------------------------
# Step 6: Visualize Cost Convergence
# ----------------------------
plt.figure(figsize=(4,3))
plt.plot(cost_history)
plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.title("Gradient Descent Convergence")
plt.show()

# ----------------------------
# Step 7: Visualize Actual vs Predicted
# ----------------------------
plt.figure(figsize=(4,4))
plt.scatter(y_test, y_pred_test, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (Gradient Descent)")
plt.show()
