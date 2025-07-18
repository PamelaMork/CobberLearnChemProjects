import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Drop rows with missing target or all features
df_numeric = df.select_dtypes(include=[np.number])
df_numeric = df_numeric.drop(columns=['alive'], errors='ignore')  # if present
original_avg_age = df_numeric['age'].mean(skipna=True)
print(f"Original average age (excluding missing): {original_avg_age:.2f}")

# Display correlation matrix
corr = df_numeric.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Titanic Dataset (Numerical Features)")
plt.show()

# Prepare data for KNN
df_knn = df_numeric.copy()
X = df_knn.drop(columns='age')
y = df_knn['age']


# Split known-age rows for model evaluation
X_known = X[y.notnull()]
y_known = y[y.notnull()]
X_missing = X[y.isnull()]

# Train KNN model on all numeric features
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_known, y_known)

# Predict ages for missing values
age_pred = knn.predict(X_missing)

# Fill in missing ages
df_knn.loc[y.isnull(), 'age'] = age_pred
new_avg_age = df_knn['age'].mean()
print(f"New average age after KNN imputation: {new_avg_age:.2f}")

# Evaluate KNN on known values
y_pred_known = knn.predict(X_known)
mae = mean_absolute_error(y_known, y_pred_known)
print(f"MAE (on known ages): {mae:.2f}")

# Plot actual vs. predicted for known ages
plt.figure(figsize=(6, 6))
plt.scatter(y_known, y_pred_known, alpha=0.5)
plt.plot([0, 80], [0, 80], 'r--')
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.title("Actual vs. Predicted Age (KNN using all numeric features)")
plt.grid(True)
plt.show()
