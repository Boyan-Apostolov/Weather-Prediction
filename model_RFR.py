import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data from a CSV file in the same directory
file_name = 'eindhoven-weather.csv'  # Replace with your actual CSV file name
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, file_name)
df = pd.read_csv(file_path)

# Convert the 'time' column to datetime
df['time'] = pd.to_datetime(df['time'])

# Drop non-numeric columns and handle missing values
df = df.drop(columns=['time'])  # Drop the time column for prediction
df = df.dropna()  # Drop rows with missing values

# Define features and target variable
X = df.drop(columns=['temperature_2m (°C)'])  # Features
y = df['temperature_2m (°C)']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.3f}')

# Visualize the results
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Temperature', marker='o')
plt.plot(y_pred, label='Predicted Temperature', marker='o')
plt.title('Actual vs Predicted Temperature')
plt.xlabel('Sample Index')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
