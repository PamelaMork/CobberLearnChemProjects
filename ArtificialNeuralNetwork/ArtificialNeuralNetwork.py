import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Build dataset
data = [
    {"Compound": "Methane", "MW": 16, "BoilingPoint": -161},
    {"Compound": "Water", "MW": 18, "BoilingPoint": 100},
    {"Compound": "Propane", "MW": 44, "BoilingPoint": -42},
    {"Compound": "Ethanol", "MW": 46, "BoilingPoint": 78},
    {"Compound": "Formic Acid", "MW": 46, "BoilingPoint": 101},
    {"Compound": "Acetic Acid", "MW": 60, "BoilingPoint": 118},
    {"Compound": "Butane", "MW": 58, "BoilingPoint": -1},
    {"Compound": "Acetone", "MW": 58, "BoilingPoint": 56},
    {"Compound": "Benzene", "MW": 78, "BoilingPoint": 80},
    {"Compound": "Toluene", "MW": 92, "BoilingPoint": 111},
    {"Compound": "Octane", "MW": 114, "BoilingPoint": 125}
]
df = pd.DataFrame(data)
X = df[["MW"]]
y = df["BoilingPoint"]

# Step 2: Train Linear Regression
linear = LinearRegression()
linear.fit(X, y)
linear_preds = linear.predict(X)

mae_lin = mean_absolute_error(y, linear_preds)
mse_lin = mean_squared_error(y, linear_preds)
r2_lin = r2_score(y, linear_preds)

print("LINEAR REGRESSION RESULTS:")
print("MAE:", round(mae_lin, 2))
print("MSE:", round(mse_lin, 2))
print("R²: ", round(r2_lin, 3))

# Plot linear regression fit
plt.figure()
plt.scatter(X["MW"], y, color="black", label="Actual Data")
plt.plot(X["MW"], linear_preds, color="blue", label="Linear Fit")
plt.title("Linear Regression Fit")
plt.xlabel("Molecular Weight")
plt.ylabel("Boiling Point")
plt.legend()
plt.savefig("linear_regression_fit.png")
plt.show()

# Plot residuals
residuals = y - linear_preds
plt.figure()
plt.scatter(X["MW"], residuals)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals: Linear Regression")
plt.xlabel("Molecular Weight")
plt.ylabel("Residual (Actual - Predicted)")
plt.savefig("linear_residuals.png")
plt.show()

# Step 3: Train Neural Network (10,10)
nn = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation="relu",
    max_iter=5000,
    early_stopping=False
)
nn.fit(X, y)
nn_preds = nn.predict(X)

mae_nn = mean_absolute_error(y, nn_preds)
mse_nn = mean_squared_error(y, nn_preds)
r2_nn = r2_score(y, nn_preds)

print("\nNEURAL NETWORK RESULTS:")
print("MAE:", round(mae_nn, 2))
print("MSE:", round(mse_nn, 2))
print("R²: ", round(r2_nn, 3))
print("Neural network training stopped after", nn.n_iter_, "epochs.")

# Plot neural network predictions
plt.figure()
plt.scatter(X["MW"], y, color="black", label="Actual Data")
plt.plot(X["MW"], nn_preds, color="green", label="NN (10,10) Prediction")
plt.title("Neural Network Fit (10,10)")
plt.xlabel("Molecular Weight")
plt.ylabel("Boiling Point")
plt.legend()
plt.savefig("nn_10_10_fit.png")
plt.show()

# Plot NN residuals
nn_residuals = y - nn_preds
plt.figure()
plt.scatter(X["MW"], nn_residuals)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals: Neural Network (10,10)")
plt.xlabel("Molecular Weight")
plt.ylabel("Residual (Actual - Predicted)")
plt.savefig("nn_10_10_residuals.png")
plt.show()

# Step 4: Comparison plot
mw_range = np.linspace(X["MW"].min(), X["MW"].max(), 100).reshape(-1, 1)
linear_line = linear.predict(mw_range)
nn_line = nn.predict(mw_range)

plt.figure()
plt.scatter(X["MW"], y, color="black", label="Actual Data")
plt.plot(mw_range, linear_line, color="blue", linestyle="--", label="Linear Regression")
plt.plot(mw_range, nn_line, color="green", linestyle="-", label="NN (10,10)")
plt.title("Comparison of Models")
plt.xlabel("Molecular Weight")
plt.ylabel("Boiling Point")
plt.legend()
plt.savefig("model_comparison_linear_vs_nn_10_10.png")
plt.show()

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Assumes you already have df, X, y, and nn (the original 10,10 model)
# Example: X = df[["MW"]], y = df["BoilingPoint"]

# Create mw_range and matching DataFrame to fix UserWarning
mw_range = np.linspace(X["MW"].min(), X["MW"].max(), 100).reshape(-1, 1)
mw_range_df = pd.DataFrame(mw_range, columns=["MW"])

# Predict once with original model for later comparison
original_nn_preds = nn.predict(mw_range_df)

# Begin interactive loop
while True:
    try:
        num_layers = int(input("How many layers do you want (1–4)? "))
        if not 1 <= num_layers <= 4:
            print("Please enter a number between 1 and 4.")
            continue

        layer_sizes = []
        for i in range(num_layers):
            neurons = int(input(f"How many neurons in layer {i + 1} (1–10)? "))
            if not 1 <= neurons <= 10:
                print("Each layer must have 1 to 10 neurons.")
                break
            layer_sizes.append(neurons)
        else:
            # Create and train the custom network
            custom_nn = MLPRegressor(
                hidden_layer_sizes=tuple(layer_sizes),
                activation='relu',
                max_iter=5000,
                early_stopping=False,
                random_state=42
            )
            custom_nn.fit(X, y)

            # Predictions and metrics
            custom_preds = custom_nn.predict(mw_range_df)
            custom_y_preds = custom_nn.predict(X)
            mae = mean_absolute_error(y, custom_y_preds)
            mse = mean_squared_error(y, custom_y_preds)
            r2 = r2_score(y, custom_y_preds)

            print(f"\nCUSTOM NEURAL NETWORK {tuple(layer_sizes)}:")
            print("MAE:", round(mae, 2))
            print("MSE:", round(mse, 2))
            print("R²: ", round(r2, 3))

            # Plot comparison
            plt.figure(figsize=(8, 5))
            plt.scatter(X["MW"], y, color="black", label="Actual Data")
            plt.plot(mw_range, original_nn_preds, color="blue", label="Original NN (10,10)")
            plt.plot(mw_range, custom_preds, color="green", linestyle="--", label=f"Custom NN {tuple(layer_sizes)}")
            plt.xlabel("Molecular Weight (MW)")
            plt.ylabel("Boiling Point (°C)")
            plt.title("Comparison of Neural Network Models")
            plt.legend()
            filename = f"NN_Comparison_{'_'.join(map(str, layer_sizes))}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()

            # Plot residuals
            residuals = y - custom_y_preds
            plt.figure(figsize=(8, 4))
            plt.scatter(X["MW"], residuals)
            plt.axhline(0, color='red', linestyle='--')
            plt.title(f"Residuals: Custom NN {tuple(layer_sizes)}")
            plt.xlabel("Molecular Weight (MW)")
            plt.ylabel("Residuals")
            plt.savefig(f"Residuals_Custom_NN_{'_'.join(map(str, layer_sizes))}.png", dpi=300, bbox_inches='tight')
            plt.show()

        # Ask to continue or stop
        again = input("\nTry another architecture? (y/n): ").strip().lower()
        if again != 'y':
            break
    except ValueError:
        print("Please enter valid integers.")
