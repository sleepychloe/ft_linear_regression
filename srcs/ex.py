import pandas as pd
import matplotlib.pyplot as plt

# Load data from a CSV file
data = pd.read_csv('data.csv')

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['km'], data['price'], color='blue', alpha=0.5)
plt.title('Car Price vs. Kilometers')
plt.xlabel('Kilometers')
plt.ylabel('Price (in USD)')
plt.grid(True)
plt.show()
