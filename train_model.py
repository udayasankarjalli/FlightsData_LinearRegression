import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Gradient Descent Linear Regression Implementation
class LinearRegressionGD:
    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.theta) + self.bias
            d_theta = -(2/m) * np.dot(X.T, (y - y_pred))
            d_bias = -(2/m) * np.sum(y - y_pred)

            self.theta -= self.lr * d_theta
            self.bias -= self.lr * d_bias

    def predict(self, X):
        return np.dot(X, self.theta) + self.bias

# 1. Load Data
data = pd.read_csv("flights_data.csv")

# 2. Handle Nulls (fill with mean)
data['flight_duration'].fillna(data['flight_duration'].mean(), inplace=True)
data['fuel_cost'].fillna(data['fuel_cost'].mean(), inplace=True)

# 3. Remove Outliers (z-score > 3)
from scipy.stats import zscore
data = data[(np.abs(zscore(data.select_dtypes(include=[np.number]))) < 3).all(axis=1)]

# 4. Features/Target
X = data.drop(columns=['price'])
y = data['price']

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Model with Gradient Descent
model = LinearRegressionGD(lr=0.001, n_iter=1000)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Save model + scaler
joblib.dump(model, "linear_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved successfully.")
