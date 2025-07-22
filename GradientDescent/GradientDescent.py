import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Generate noisy linear data
np.random.seed(42)
x = np.linspace(0, 10, 11)
true_slope = 2
true_intercept = 5
noise = np.random.normal(0, 2, size=x.shape)
y = true_slope * x + true_intercept + noise

# Step 2: Train a linear regression model
X = x.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Step 3: Print the learned slope and intercept
print(f"Learned slope (model.coef_): {model.coef_[0]:.2f}")
print(f"Learned intercept (model.intercept_): {model.intercept_:.2f}")

# Step 4: Plot the noisy data, true line, and model prediction
plt.scatter(x, y, color='blue', label='Noisy data')
plt.plot(x, true_slope * x + true_intercept, 'r--', label='True line')
plt.plot(x, y_pred, 'g-', label='Model prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Define the MSE function
def compute_mse(m, b, x, y):
    predictions = m * x + b
    errors = predictions - y
    return np.mean(errors ** 2)

# Step 6: Ask the student to enter a slope and intercept, then calculate MSE
try:
    m_input = float(input("Enter a slope (m): "))
    b_input = float(input("Enter an intercept (b): "))
    mse = compute_mse(m_input, b_input, x, y)
    print(f"MSE for your guess (m = {m_input}, b = {b_input}): {mse:.2f}")
except ValueError:
    print("Please enter valid numbers for slope and intercept.")

# Step 7: With the help of your assistant, write Python code that builds a grid of slope and intercept values
# and calculates the MSE at each point to create a loss landscape for linear regression.
# Low MSE values should show up as yellow and high values as purple.

m_values = np.linspace(-1, 5, 100)
b_values = np.linspace(0, 10, 100)
M, B = np.meshgrid(m_values, b_values)

Z = np.zeros_like(M)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        Z[i, j] = compute_mse(M[i, j], B[i, j], x, y)

# Plot the loss landscape with yellow = low MSE, purple = high MSE
plt.figure(figsize=(8, 6))
contour = plt.contourf(M, B, Z, levels=50, cmap='plasma_r')  # reversed colormap
plt.colorbar(contour, label='MSE Loss')
plt.xlabel('Slope (m)')
plt.ylabel('Intercept (b)')
plt.title('Loss Landscape for Linear Regression')

# Add markers for true parameters and model prediction
plt.plot(true_slope, true_intercept, 'g*', markersize=12, label='True parameters')
plt.plot(model.coef_[0], model.intercept_, 'bx', markersize=10, label='Model solution')
plt.legend()
plt.grid(True)
plt.show()
