import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Step 1: Generate noisy quadratic data
np.random.seed(42)
x = np.linspace(0, 10, 11)
true_a = 1
true_b = 2
true_c = 5
noise = np.random.normal(0, 2, size=x.shape)
y = true_a * x**2 + true_b * x + true_c + noise

# Reshape x for sklearn
X = x.reshape(-1, 1)

# Step 2: Polynomial transformation (degree 2)
degree = 2
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)

# Step 3: Fit the polynomial model
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

# Step 4: Print learned coefficients
coeffs = model.coef_
intercept = model.intercept_
print(f"Learned coefficients (degree {degree}): {coeffs}")
print(f"Intercept: {intercept:.2f}")

# Step 5: Plot the polynomial fit
plt.scatter(x, y, color='blue', label='Noisy data')
x_smooth = np.linspace(0, 10, 200).reshape(-1, 1)
x_smooth_poly = poly.transform(x_smooth)
y_smooth_pred = model.predict(x_smooth_poly)
plt.plot(x_smooth, y_smooth_pred, color='purple', label=f'Polynomial Fit (degree 2)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression Fit')
plt.legend()
plt.grid(True)
plt.show()

# Step 6: MSE function for polynomial model
def compute_poly_mse(a, b, c, x, y):
    predictions = a * x**2 + b * x + c
    errors = predictions - y
    return np.mean(errors ** 2)

# Step 7: Let student guess polynomial coefficients with suggested ranges
try:
    a_input = float(input("Enter coefficient for x² (a), suggested range 0 to 2: "))
    while not (0 <= a_input <= 2):
        a_input = float(input("Please enter a value between 0 and 2 for a: "))

    b_input = float(input("Enter coefficient for x (b), suggested range 0 to 4: "))
    while not (0 <= b_input <= 4):
        b_input = float(input("Please enter a value between 0 and 4 for b: "))

    c_input = float(input("Enter intercept (c), suggested range 0 to 10: "))
    while not (0 <= c_input <= 10):
        c_input = float(input("Please enter a value between 0 and 10 for c: "))

    mse = compute_poly_mse(a_input, b_input, c_input, x, y)
    print(f"MSE for your guess (a = {a_input}, b = {b_input}, c = {c_input}): {mse:.2f}")

except ValueError:
    print("Please enter valid numeric values.")

# Step 8: Build a loss landscape (c fixed at 5)
a_vals = np.linspace(0, 2, 100)
b_vals = np.linspace(0, 4, 100)
A, B = np.meshgrid(a_vals, b_vals)
Z = np.zeros_like(A)
fixed_c = 5

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        Z[i, j] = compute_poly_mse(A[i, j], B[i, j], fixed_c, x, y)

# Step 9: Gradient descent setup
def numerical_gradient(a, b, c, x, y, epsilon=1e-3):
    da = (compute_poly_mse(a + epsilon, b, c, x, y) - compute_poly_mse(a - epsilon, b, c, x, y)) / (2 * epsilon)
    db = (compute_poly_mse(a, b + epsilon, c, x, y) - compute_poly_mse(a, b - epsilon, c, x, y)) / (2 * epsilon)
    return da, db

# Gradient descent parameters
a_path = []
b_path = []
loss_path = []

a_curr = 1.0   # Start near the true value
b_curr = 2.0
c_fixed = fixed_c
learning_rate = 0.005
num_steps = 50

for step in range(num_steps):
    loss = compute_poly_mse(a_curr, b_curr, c_fixed, x, y)
    da, db = numerical_gradient(a_curr, b_curr, c_fixed, x, y)

    a_path.append(a_curr)
    b_path.append(b_curr)
    loss_path.append(loss)

    a_curr -= learning_rate * da
    b_curr -= learning_rate * db

    # Keep parameters in safe bounds
    a_curr = np.clip(a_curr, 0, 2)
    b_curr = np.clip(b_curr, 0, 4)

# Step 10: Plot loss landscape and descent path
plt.figure(figsize=(8, 6))
contour = plt.contourf(
    A, B, Z,
    levels=100,
    cmap='plasma_r',
    vmin=Z.min(),
    vmax=Z.max()
)
plt.colorbar(contour, label='MSE Loss')
plt.xlabel('Coefficient a (for x²)')
plt.ylabel('Coefficient b (for x)')
plt.title('Loss Landscape with Gradient Descent Path')

# True parameters and model solution
plt.plot(true_a, true_b, 'g*', markersize=12, label='True parameters')
plt.plot(coeffs[2], coeffs[1], 'bx', markersize=10, label='Model solution')

# Gradient descent path
plt.plot(a_path, b_path, 'r.-', label='Gradient Descent Path')

plt.legend()
plt.grid(True)
plt.show()
