import pandas as pd
import numpy as np

# 1. Generate Synthetic Data
np.random.seed(42)
num_samples = 1000

# Create features
flight_duration = np.random.uniform(1, 12, num_samples)
distance = flight_duration * 500 + np.random.normal(0, 100, num_samples)
seats_booked = np.random.randint(50, 200, num_samples)
fuel_cost = distance * np.random.uniform(0.5, 1.5, num_samples) + np.random.normal(0, 50, num_samples)
airline_rating = np.random.randint(1, 11, num_samples)

# Target variable 'price'
price = (
    (20 * flight_duration)
    + (0.01 * distance)
    + (1.5 * seats_booked)
    + (0.05 * fuel_cost)
    - (5 * airline_rating)
    + np.random.normal(0, 50, num_samples)
)

# Create DataFrame
data = pd.DataFrame({
    'flight_duration': flight_duration,
    'distance': distance,
    'seats_booked': seats_booked,
    'fuel_cost': fuel_cost,
    'airline_rating': airline_rating,
    'price': price
})

# 2. Introduce Nulls
data.loc[data.sample(frac=0.05).index, 'flight_duration'] = np.nan
data.loc[data.sample(frac=0.07).index, 'fuel_cost'] = np.nan

# 3. Introduce Outliers
data.loc[data.sample(frac=0.02).index, 'price'] = np.random.uniform(1000, 5000, int(num_samples * 0.02))

# Save dataset
data.to_csv("flights_data.csv", index=False)
print("Synthetic flight dataset saved to flights_data.csv")
