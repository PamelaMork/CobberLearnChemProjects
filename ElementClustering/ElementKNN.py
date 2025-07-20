import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.metrics import pairwise_distances

# Load the data
df = pd.read_csv('group1_elements.csv')

# Step 1: Create labels for classification
df['Label'] = df['Atomic Number'].apply(lambda x: 'light' if x <= 19 else 'heavy')

# Step 2: Convert features and labels
features = df[['Atomic Radius (pm)', 'First Ionization Energy (kJ/mol)']].values
labels = df['Label'].values
symbols = df['Symbol'].values

# Color map
label_colors = {'light': 'skyblue', 'heavy': 'salmon'}

# Set value of k
k = 3

# Step 3: Loop through each element as a test case
for test_index in range(len(df)):
    # Set the test point
    test_point = features[test_index]
    test_label = labels[test_index]
    test_symbol = symbols[test_index]

    # Use the rest as training data
    train_features = np.delete(features, test_index, axis=0)
    train_labels = np.delete(labels, test_index, axis=0)
    train_symbols = np.delete(symbols, test_index, axis=0)

    # Compute distances from test point to all training points
    distances = pairwise_distances([test_point], train_features)[0]

    # Get indices of k nearest neighbors
    nearest_indices = distances.argsort()[:k]
    nearest_labels = train_labels[nearest_indices]

    # Majority vote
    unique, counts = np.unique(nearest_labels, return_counts=True)
    predicted_label = unique[np.argmax(counts)]

    # Step 4: Plot
    plt.figure(figsize=(8, 6))

    # Plot training points
    for i, (x, y) in enumerate(train_features):
        label = train_labels[i]
        plt.scatter(x, y, color=label_colors[label], edgecolor='black', s=100)
        plt.text(x + 3, y, train_symbols[i], fontsize=10)

    # Plot test point as X
    plt.scatter(test_point[0], test_point[1], color=label_colors[predicted_label],
                edgecolor='black', marker='X', s=200, label='Test Point')

    # Highlight the k nearest neighbors
    for i in nearest_indices:
        neighbor = train_features[i]
        plt.plot([test_point[0], neighbor[0]], [test_point[1], neighbor[1]],
                 color='gray', linestyle='dotted')

    # Labels
    plt.title(f"Predicting: {test_symbol}  →  Predicted: {predicted_label.upper()}", fontsize=14)
    plt.xlabel('Atomic Radius (pm)')
    plt.ylabel('First Ionization Energy (kJ/mol)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    input("Press Enter to see the next element...")
