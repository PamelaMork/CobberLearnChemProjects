import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Step 1: Create the DataFrame
data = {
    'Molecule': ['Molecule 1', 'Molecule 2', 'Molecule 3', 'Molecule 4',
                 'Molecule 5', 'Molecule 6', 'Molecule 7', 'Molecule 8'],
    'Molecular Weight': [180, 250, 80, 300, 150, 400, 90, 200],
    'Hydrogen Bond Donors': [5, 2, 1, 1, 4, 3, 0, 2],
    'Hydrogen Bond Acceptors': [6, 3, 2, 2, 5, 4, 1, 3],
    'Water Solubility': [1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
print(df)

# Step 2: Split into features and target
X = df[['Molecular Weight', 'Hydrogen Bond Donors', 'Hydrogen Bond Acceptors']]
y = df['Water Solubility']

# Step 3: Train the decision tree
model = DecisionTreeClassifier()
model.fit(X, y)

# Step 4: Visualize the tree
plt.figure(figsize=(10, 6))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Not Soluble', 'Soluble'],
    filled=True,
    rounded=True
)
plt.show()
