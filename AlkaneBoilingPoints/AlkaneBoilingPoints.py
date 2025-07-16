import matplotlib.pyplot as plt

# --- Data for the first 10 straight-chain (normal) alkanes ---
carbons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]          # number of carbon atoms
boiling_points = [                                 # boiling points in °C
    -161.5,  # methane
    -88.6,   # ethane
    -42.1,   # propane
    -0.5,    # butane
     36.1,   # pentane
     68.7,   # hexane
     98.4,   # heptane
    125.6,   # octane
    150.8,   # nonane
    174.0    # decane
]

# --- Create the scatter plot ---
plt.scatter(carbons, boiling_points)

# --- Add title and axis labels ---
plt.title("Boiling Point vs Number of Carbons (First 10 Linear Alkanes)")
plt.xlabel("Number of Carbons")
plt.ylabel("Boiling Point (°C)")

# --- Display the plot ---
plt.show()