import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from a CSV file in the same directory
file_name = 'eindhoven-weather.csv'  # Replace with your actual CSV file name
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, file_name)
df = pd.read_csv(file_path)

# Convert the 'time' column to datetime
df['time'] = pd.to_datetime(df['time'])

# Drop non-numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=[np.number])

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".3f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap')
plt.show()
