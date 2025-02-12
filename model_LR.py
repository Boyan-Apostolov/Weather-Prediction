import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import matplotlib.pyplot as plt

# Load the CSV file
file_name = 'eindhoven-weather.csv'  # Replace with your actual CSV file name
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, file_name)
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Preprocess the data
# Convert 'time' to datetime format
data['time'] = pd.to_datetime(data['time'])

# Define features (X) and target (y)
X = data.drop(['time', 'temperature_2m (°C)'], axis=1)
y = data['temperature_2m (°C)']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(X, y, data['time'], test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Plotting the real data vs predictions over time
plt.figure(figsize=(12, 6))

# Plot the actual temperatures
plt.plot(time_test, y_test, color='blue', marker='o', label='Actual Temperature', alpha=0.6)

# Plot the predicted temperatures
plt.plot(time_test, y_pred, color='orange', marker='x', label='Predicted Temperature', alpha=0.6)

# Formatting the plot
plt.title('Real vs Predicted Temperature Over Time')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Example of making a prediction
# Create a DataFrame for the sample input
sample_input = pd.DataFrame({
    'relative_humidity_2m (%)': [94],
    'precipitation (mm)': [0.00],
    'rain (mm)': [0.00],
    'pressure_msl (hPa)': [1034.8],
    'wind_speed_10m (km/h)': [10.0]
})

# Make the prediction
predicted_temperature = model.predict(sample_input)
print(f'Predicted Temperature: {predicted_temperature[0]} °C')
