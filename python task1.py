import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("sales_data.csv")

# Convert months to numbers
data['Month_num'] = range(1, len(data)+1)

# Training data
X = data[['Month_num']]
y = data['Sales']

# Model
model = LinearRegression()
model.fit(X, y)

# Predict next 3 months
future_months = [[13], [14], [15]]
predictions = model.predict(future_months)

print("Future Predictions:", predictions)

# Plot
plt.scatter(data['Month_num'], y, color='blue')
plt.plot(data['Month_num'], model.predict(X), color='red')
plt.title("Sales Forecast")
plt.xlabel("Month Number")
plt.ylabel("Sales")

plt.show()