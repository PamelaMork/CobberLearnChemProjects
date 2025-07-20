import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the CSV file once
csv_filename = 'group1_elements.csv'
df_original = pd.read_csv(csv_filename)

# Prepare color palette
colors = ['tomato', 'royalblue', 'seagreen', 'gold', 'orchid', 'slateblue', 'darkorange', 'crimson']


def plot_clusters(k):
    # Make a copy of the original DataFrame
    df = df_original.copy()

    # Select features
    X = df[['Atomic Radius (pm)', 'First Ionization Energy (kJ/mol)']]

    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    df['Color'] = df['Cluster'].map(lambda i: colors[i % len(colors)])

    # Get cluster centers
    centers = kmeans.cluster_centers_

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Atomic Radius (pm)'], df['First Ionization Energy (kJ/mol)'],
                c=df['Color'], edgecolor='black', s=100, label='Elements')

    # Label each point with the element’s symbol
    for _, row in df.iterrows():
        plt.text(row['Atomic Radius (pm)'] + 3,
                 row['First Ionization Energy (kJ/mol)'],
                 row['Symbol'], fontsize=10)

    # Plot cluster centers (without labels)
    plt.scatter(centers[:, 0], centers[:, 1],
                c='black', s=200, marker='X', label='Cluster Centers')

    # Plot styling
    plt.title(f'K-Means Clustering of Group 1 Elements (k = {k})', fontsize=14)
    plt.xlabel('Atomic Radius (pm)', fontsize=12)
    plt.ylabel('First Ionization Energy (kJ/mol)', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Interactive loop
while True:
    try:
        k = int(input("Enter the number of clusters (k): "))
        plot_clusters(k)
    except ValueError:
        print("Please enter a valid integer.")
        continue

    again = input("Would you like to try another value of k? (yes/no): ").strip().lower()
    if again != 'yes':
        print("Goodbye!")
        break
