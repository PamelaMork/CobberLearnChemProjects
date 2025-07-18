import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define arrays
actual = np.array([2, 4, 5, 4, 5, 7, 9])
predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])
residuals = predicted - actual

# Create DataFrame
results = pd.DataFrame({
    'Actual': actual,
    'Predicted': predicted,
    'Residual': residuals
})

# Calculate metrics
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
r2 = r2_score(actual, predicted)

# Print table and metrics
print(results)
print(f"\nMAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

# Find the worst prediction
worst_index = np.argmax(np.abs(residuals))

# Plot: Actual vs. Predicted with the worst point highlighted
plt.figure(figsize=(8, 5))
plt.scatter(actual, predicted, color='teal', edgecolor='black', s=100)
plt.plot([min(actual), max(actual)], [min(actual), max(actual)],
         color='black', linestyle='--', label='Perfect Prediction')
plt.scatter(actual[worst_index], predicted[worst_index],
            color='red', s=150, edgecolor='black', label='Worst Prediction')

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted with Worst Prediction Highlighted")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
